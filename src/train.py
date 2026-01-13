from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed as accel_set_seed
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from .config import TrainConfig
from .utils import append_jsonl, ensure_dir, read_jsonl, write_json


# -----------------------------
# LoRA helpers
# -----------------------------
def _infer_lora_target_modules(model: torch.nn.Module) -> List[str]:
    """
    Automatically determine which transformer modules to apply LoRA to.

    Targets the standard attention and MLP modules that benefit most from
    parameter-efficient fine-tuning. Falls back to all linear layers if
    preferred modules are not found.

    Args:
        model: Transformer model to analyze

    Returns:
        List of module names to apply LoRA to
    """
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    present: set[str] = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in preferred and isinstance(module, torch.nn.Linear):
            present.add(leaf)

    if present:
        return [m for m in preferred if m in present]

    return ["all-linear"]


# -----------------------------
# Dataset building
# -----------------------------
def _make_dataset(
    rows: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    min_target_tokens: int = 32,
) -> List[Dict[str, torch.Tensor]]:
    """
    Convert raw text examples to tokenized tensors for training.

    Applies length truncation to fit within max_length while preserving
    minimum target tokens. Adds EOS token to targets if missing.

    Args:
        rows: List of example dicts with 'prompt' and 'target_text' keys
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        min_target_tokens: Minimum tokens to preserve in target (default: 32)

    Returns:
        List of dicts with 'input_ids', 'attention_mask', 'labels' tensors
    """
    data: List[Dict[str, torch.Tensor]] = []
    eos_id = tokenizer.eos_token_id

    for r in rows:
        prompt = str(r["prompt"])
        target = str(r["target_text"]).strip()
        if not target:
            continue

        prompt_ids: List[int] = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids: List[int] = tokenizer.encode(target, add_special_tokens=False)

        if eos_id is not None and (len(target_ids) == 0 or target_ids[-1] != eos_id):
            target_ids = target_ids + [eos_id]

        if len(prompt_ids) + len(target_ids) > max_length:
            min_t = min(min_target_tokens, len(target_ids))
            budget_for_prompt = max_length - min_t
            prompt_ids = prompt_ids[: max(0, budget_for_prompt)]
            target_ids = target_ids[: max_length - len(prompt_ids)]

        ids = prompt_ids + target_ids
        input_ids = torch.tensor(ids, dtype=torch.long)

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100

        data.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

    if not data:
        raise RuntimeError("No training examples were created. Check your input JSONL and filtering.")
    return data


def _collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Collate a list of samples into padded batch tensors.
    Works both for normal DataLoader batching and for batch_sampler batching.
    """
    def pad_1d(x_list: Sequence[torch.Tensor], pad_val: int) -> torch.Tensor:
        max_len = max(x.size(0) for x in x_list)
        out = torch.full((len(x_list), max_len), pad_val, dtype=x_list[0].dtype)
        for i, x in enumerate(x_list):
            out[i, : x.size(0)] = x
        return out

    input_ids = pad_1d([b["input_ids"] for b in batch], pad_token_id)
    attention_mask = pad_1d([b["attention_mask"] for b in batch], 0)
    labels = pad_1d([b["labels"] for b in batch], -100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _count_trainable_params(model: torch.nn.Module) -> Dict[str, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"trainable": trainable, "total": total}


# -----------------------------
# Length-bucketing
# -----------------------------
class BucketBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups examples by similar length to reduce padding.

    DDP friendliness:
      - yields batches (list[int])
      - Accelerate will shard batches across ranks
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        bucket_size: int = 32,
        drop_last: bool = False,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if bucket_size <= 0:
            raise ValueError("bucket_size must be > 0")

        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.drop_last = drop_last
        self.seed = seed
        self.shuffle = shuffle

        self.epoch = 0
        self._batches: List[List[int]] = []
        self._rebuild()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._rebuild()

    def _rebuild(self) -> None:
        buckets: Dict[int, List[int]] = {}
        for i, L in enumerate(self.lengths):
            bid = int(L) // self.bucket_size
            buckets.setdefault(bid, []).append(i)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        batches: List[List[int]] = []
        for _, idxs in buckets.items():
            if self.shuffle:
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[j] for j in perm]

            for j in range(0, len(idxs), self.batch_size):
                b = idxs[j : j + self.batch_size]
                if self.drop_last and len(b) < self.batch_size:
                    continue
                batches.append(b)

        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[j] for j in perm]

        self._batches = batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


# -----------------------------
# Checkpoint utilities
# -----------------------------
def _is_checkpoint_dir(p: str | Path) -> bool:
    path = Path(p)
    return path.exists() and path.is_dir() and (path / "trainer_state.json").exists()


def _load_trainer_state(ckpt_dir: Path) -> Dict[str, Any]:
    with open(ckpt_dir / "trainer_state.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _save_trainer_state(ckpt_dir: Path, state: Dict[str, Any]) -> None:
    with open(ckpt_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# -----------------------------
# Training helpers
# -----------------------------
def _setup_training_environment(cfg: TrainConfig) -> tuple[Accelerator, SummaryWriter | None, Path]:
    """Set up accelerator, logging, and output directories."""
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum,
        mixed_precision=None if cfg.mixed_precision == "no" else cfg.mixed_precision,
        log_with=None,
    )

    accel_set_seed(cfg.seed)

    out_dir = ensure_dir(cfg.out_dir)
    ensure_dir(out_dir / cfg.checkpoints_dir)

    writer: SummaryWriter | None = None
    if accelerator.is_main_process and cfg.tensorboard:
        tb_path = ensure_dir(out_dir / cfg.tb_dir)
        writer = SummaryWriter(log_dir=str(tb_path))

    return accelerator, writer, out_dir


def _load_and_preprocess_data(cfg: TrainConfig) -> tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], Any]:
    """Load and preprocess training/validation data."""
    # Load data
    rows_train = read_jsonl(cfg.train_path)
    rows_val = read_jsonl(cfg.val_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocess data
    train_data = _make_dataset(rows_train, tokenizer, cfg.max_length)
    val_data = _make_dataset(rows_val, tokenizer, cfg.max_length)

    return train_data, val_data, tokenizer


def _setup_model_and_training(
    cfg: TrainConfig,
    train_data: List[Dict[str, torch.Tensor]],
    val_data: List[Dict[str, torch.Tensor]],
    tokenizer: Any,
    accelerator: Accelerator
) -> tuple[Any, Any, Any, Any, Any, List[int]]:
    """Set up model, LoRA, optimizer, scheduler, and dataloaders."""
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    if hasattr(base, "config"):
        base.config.use_cache = False

    target_modules = _infer_lora_target_modules(base)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(base, lora)
    model.train()

    # Build datasets
    train_lengths = [int(ex["input_ids"].numel()) for ex in train_data]

    # Bucketing knobs
    bucket_by_length = bool(getattr(cfg, "bucket_by_length", True))
    bucket_size = int(getattr(cfg, "bucket_size", 32))
    bucket_shuffle = bool(getattr(cfg, "bucket_shuffle", True))

    pin = torch.cuda.is_available()

    # Preserve sampler reference for set_epoch calls after accelerator.prepare
    train_bucket_sampler: BucketBatchSampler | None = None
    if bucket_by_length:
        train_bucket_sampler = BucketBatchSampler(
            lengths=train_lengths,
            batch_size=cfg.batch_size,
            bucket_size=bucket_size,
            drop_last=False,
            seed=cfg.seed,
            shuffle=bucket_shuffle,
        )

    # If using batch_sampler, do NOT set batch_size/shuffle
    dl_train = DataLoader(
        train_data,
        batch_size=cfg.batch_size if train_bucket_sampler is None else 1,  # ignored if batch_sampler is set
        shuffle=(train_bucket_sampler is None),
        batch_sampler=train_bucket_sampler,
        pin_memory=pin,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=(lambda batch: _collate(batch, tokenizer.pad_token_id)),
    )

    dl_val = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=(lambda batch: _collate(batch, tokenizer.pad_token_id)),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Compute schedule steps BEFORE prepare (so it reflects the pre-shard batch count)
    steps_per_epoch_micro_pre = len(dl_train)
    steps_per_epoch_optim_pre = math.ceil(steps_per_epoch_micro_pre / cfg.grad_accum)
    total_optim_steps = steps_per_epoch_optim_pre * cfg.epochs

    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(cfg.warmup_ratio * total_optim_steps),
        num_training_steps=total_optim_steps,
    )

    # Prepare for DDP
    model, optim, dl_train, dl_val, sched = accelerator.prepare(model, optim, dl_train, dl_val, sched)

    return model, optim, sched, dl_train, dl_val, [steps_per_epoch_micro_pre, steps_per_epoch_optim_pre, total_optim_steps]


def _run_training_loop(
    cfg: TrainConfig,
    accelerator: Accelerator,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    dataloaders: tuple,
    writer: SummaryWriter | None,
    out_dir: Path,
    steps_info: List[int]
) -> Path:
    """Run the main training and validation loop."""
    dl_train, dl_val = dataloaders
    steps_per_epoch_micro_pre, steps_per_epoch_optim_pre, total_optim_steps = steps_info

    # Run config + eval history init
    if accelerator.is_main_process:
        run_cfg = asdict(cfg)
        run_cfg["lora_target_modules"] = _infer_lora_target_modules(accelerator.unwrap_model(model))
        run_cfg["param_counts"] = _count_trainable_params(accelerator.unwrap_model(model))
        run_cfg["world_size"] = accelerator.num_processes
        run_cfg["effective_batch_size"] = cfg.batch_size * cfg.grad_accum * accelerator.num_processes
        run_cfg["train_rows"] = len(read_jsonl(cfg.train_path))
        run_cfg["val_rows"] = len(read_jsonl(cfg.val_path))
        run_cfg["steps_per_epoch_micro_pre"] = steps_per_epoch_micro_pre
        run_cfg["steps_per_epoch_optim_pre"] = steps_per_epoch_optim_pre
        run_cfg["total_optim_steps"] = total_optim_steps
        run_cfg["bucketing"] = {"enabled": getattr(cfg, "bucket_by_length", True), "bucket_size": getattr(cfg, "bucket_size", 32), "shuffle": getattr(cfg, "bucket_shuffle", True)}
        write_json(out_dir / "run_config.json", run_cfg)

        eval_history_path = out_dir / cfg.eval_history_file
        if not eval_history_path.exists():
            append_jsonl(
                eval_history_path,
                {
                    "event": "header",
                    "created_at_unix": time.time(),
                    "notes": "Each subsequent line is one epoch summary.",
                },
            )

    # Training state
    best_val = float("inf")
    global_step = 0
    start_epoch = 0

    # Resume
    if cfg.resume_from:
        ckpt_dir = Path(cfg.resume_from)
        if not _is_checkpoint_dir(ckpt_dir):
            raise ValueError(f"--resume-from does not look like a valid checkpoint dir: {ckpt_dir}")

        accelerator.print("[resume] Checkpoint loaded, continuing training")
        accelerator.load_state(str(ckpt_dir))

    best_dir = out_dir / "best"
    latest_dir = out_dir / "latest"
    ckpt_root = out_dir / cfg.checkpoints_dir

    if accelerator.is_main_process:
        best_dir.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(name: str, epoch: int, val_loss: float) -> None:
        if not accelerator.is_main_process:
            return
        ckpt_dir = ckpt_root / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        accelerator.save_state(str(ckpt_dir))
        _save_trainer_state(
            ckpt_dir,
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "val_loss": val_loss,
                "saved_at_unix": time.time(),
            },
        )

    # ---- Training loop ----
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        model.train()

        # Deterministic reshuffle per-epoch
        if hasattr(dl_train, 'batch_sampler') and hasattr(dl_train.batch_sampler, 'set_epoch'):
            dl_train.batch_sampler.set_epoch(epoch)

        pbar = tqdm(
            dl_train,
            desc=f"train epoch {epoch+1}/{cfg.epochs}",
            disable=not accelerator.is_main_process,
        )

        train_loss_sum = 0.0
        train_loss_count = 0

        # Throughput counters
        token_count = 0
        micro_steps = 0

        for _, batch in enumerate(pbar, start=1):
            micro_steps += 1
            # approximate tokens processed (sum of attention masks)
            if "attention_mask" in batch:
                token_count += int(batch["attention_mask"].sum().item())

            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)

                train_loss_sum += float(loss.detach().item())
                train_loss_count += 1

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % max(1, cfg.log_every_steps) == 0:
                            lr = float(scheduler.get_last_lr()[0]) if scheduler is not None else float(cfg.lr)
                            if writer is not None:
                                writer.add_scalar("train/loss_micro", float(loss.detach().item()), global_step)
                                writer.add_scalar("train/lr", lr, global_step)
                            pbar.set_postfix(loss=f"{float(loss.detach().item()):.4f}")

                        if cfg.save_every_epochs and ((epoch + 1) % cfg.save_every_epochs == 0):
                            save_checkpoint(name=f"epoch_{epoch+1}", epoch=epoch, val_loss=float("nan"))

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in dl_val:
                out = model(**batch)
                val_loss_sum += float(out.loss.detach().item())
                val_batches += 1

        val_loss_tensor = torch.tensor([val_loss_sum, val_batches], device=accelerator.device, dtype=torch.float32)
        val_loss_tensor = accelerator.reduce(val_loss_tensor, reduction="sum")
        val_loss_sum_all = float(val_loss_tensor[0].item())
        val_batches_all = float(val_loss_tensor[1].item())
        val_loss = val_loss_sum_all / max(val_batches_all, 1.0)

        train_loss_epoch = train_loss_sum / max(train_loss_count, 1)
        elapsed = time.time() - t0

        # Throughput
        tok_per_sec = token_count / max(elapsed, 1e-6)
        micro_per_sec = micro_steps / max(elapsed, 1e-6)

        if accelerator.is_main_process and writer is not None:
            writer.add_scalar("train/loss_epoch_microavg", train_loss_epoch, epoch + 1)
            writer.add_scalar("val/loss", val_loss, epoch + 1)
            writer.add_scalar("perf/tokens_per_sec", tok_per_sec, epoch + 1)
            writer.add_scalar("perf/micro_steps_per_sec", micro_per_sec, epoch + 1)

        # Save adapters
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)

            latest_dir.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(latest_dir)
            # Save tokenizer to latest
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(latest_dir)

            if val_loss < best_val:
                best_val = val_loss
                best_dir.mkdir(parents=True, exist_ok=True)
                unwrapped.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

        if cfg.save_every_epochs and ((epoch + 1) % cfg.save_every_epochs == 0):
            save_checkpoint(name=f"epoch_{epoch+1}", epoch=epoch, val_loss=val_loss)

        if accelerator.is_main_process:
            lr_now = float(scheduler.get_last_lr()[0]) if scheduler is not None else float(cfg.lr)
            eval_history_path = out_dir / cfg.eval_history_file
            append_jsonl(
                eval_history_path,
                {
                    "event": "epoch_end",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss_epoch_microavg": train_loss_epoch,
                    "val_loss": val_loss,
                    "lr": lr_now,
                    "best_val_loss_so_far": best_val,
                    "seconds": elapsed,
                    "world_size": accelerator.num_processes,
                    "effective_batch_size": cfg.batch_size * cfg.grad_accum * accelerator.num_processes,
                    "seed": cfg.seed,
                    "bucketing": {"enabled": getattr(cfg, "bucket_by_length", True), "bucket_size": getattr(cfg, "bucket_size", 32), "shuffle": getattr(cfg, "bucket_shuffle", True)},
                    "perf": {"tokens_per_sec": tok_per_sec, "micro_steps_per_sec": micro_per_sec},
                },
            )

        accelerator.wait_for_everyone()

    if writer is not None:
        writer.flush()
        writer.close()

    return best_dir


# -----------------------------
# Main train
# -----------------------------
def train_lora(cfg: TrainConfig) -> Path:
    """
    Fine-tune a language model with LoRA for refusal supervision.

    Loads base model, applies LoRA adapters, trains on refusal-aware data,
    and saves the best-performing adapter based on validation loss.

    Args:
        cfg: Training configuration dataclass

    Returns:
        Path to the best model adapter directory

    Note:
        Uses HuggingFace Accelerate for distributed training support.
        Automatically infers optimal LoRA target modules from model architecture.
    """
    # Setup phase
    accelerator, writer, out_dir = _setup_training_environment(cfg)

    # Data phase
    train_data, val_data, tokenizer = _load_and_preprocess_data(cfg)

    # Model phase
    model, optimizer, scheduler, dl_train, dl_val, steps_info = _setup_model_and_training(
        cfg, train_data, val_data, tokenizer, accelerator
    )

    # Training phase
    best_model_path = _run_training_loop(
        cfg, accelerator, model, optimizer, scheduler, (dl_train, dl_val), writer, out_dir, steps_info
    )

    return best_model_path
