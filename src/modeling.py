from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any


def _is_lora_adapter_dir(p: str | Path) -> bool:
    path = Path(p)
    return path.exists() and path.is_dir() and (path / "adapter_config.json").exists()


def _load_base_name_from_adapter(adapter_dir: str | Path) -> str:
    """
    adapter_config.json contains the base model under either:
      - "base_model_name_or_path" (most common)
      - sometimes "model_name_or_path"
    """
    import json

    cfg_path = Path(adapter_dir) / "adapter_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base = cfg.get("base_model_name_or_path") or cfg.get("model_name_or_path")
    if not base or not isinstance(base, str):
        raise ValueError(f"Could not find base model name in {cfg_path}")
    return base


def load_text_model(model_name: str) -> ModelBundle:
    """
    Loads either:
      (A) a normal HF CausalLM model id/path, OR
      (B) a PEFT LoRA adapter directory saved by model.save_pretrained()

    If model_name is an adapter dir, we load base + apply adapter.
    """
    is_adapter = _is_lora_adapter_dir(model_name)

    if is_adapter and PeftModel is None:
        raise RuntimeError(
            "Detected a LoRA adapter directory, but peft is not available. "
            "Install peft and try again."
        )

    # Prefer tokenizer saved alongside adapter, otherwise fall back to base tokenizer.
    if is_adapter:
        base_name = _load_base_name_from_adapter(model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    else:
        base_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token exists for batching
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left-pad to avoid attention on padding tokens during generation
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if is_adapter:
        model = PeftModel.from_pretrained(base_model, model_name)
    else:
        model = base_model

    model.eval()
    return ModelBundle(model=model, tokenizer=tokenizer)


@torch.inference_mode()
def generate_text(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int = 64,
    max_length: int = 1024,
) -> str:
    return generate_text_batch(
        bundle=bundle,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        max_length=max_length,
    )[0]


@torch.inference_mode()
def generate_text_batch(
    bundle: ModelBundle,
    prompts: List[str],
    max_new_tokens: int = 64,
    max_length: int = 1024,
) -> List[str]:
    """
    Batched greedy generation. Returns ONLY the continuation after each prompt.

    Notes:
      - Uses left padding (set in load_text_model) for decoder-only correctness.
      - Extracts continuations via token slicing.
    """
    tok = bundle.tokenizer

    inputs = tok(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {k: v.to(bundle.model.device) for k, v in inputs.items()}

    # Prompt lengths (in tokens) per example
    prompt_lens = inputs["attention_mask"].sum(dim=1).to(torch.long)

    # Autocast helps throughput on modern NVIDIA GPUs when model uses bf16/fp16
    use_amp = torch.cuda.is_available() and bundle.model.dtype in (torch.bfloat16, torch.float16)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    if use_amp:
        with torch.autocast(device_type="cuda", dtype=bundle.model.dtype):
            out = bundle.model.generate(**inputs, **gen_kwargs)
    else:
        out = bundle.model.generate(**inputs, **gen_kwargs)

    # Token-based continuation extraction
    results: List[str] = []
    for i in range(out.size(0)):
        pl = int(prompt_lens[i].item())
        cont_ids = out[i, pl:]
        results.append(tok.decode(cont_ids, skip_special_tokens=True).strip())

    return results
