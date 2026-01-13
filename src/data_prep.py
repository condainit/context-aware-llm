from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm

from .config import REFUSAL_TOKEN
from .prompts import build_prompt
from .utils import ensure_dir, write_json, write_jsonl


@dataclass
class SplitSizes:
    train_size: int
    val_size: int
    test_size: int


def _safe_get_question_text(example: Dict[str, Any]) -> str:
    q = example.get("question")
    if isinstance(q, dict):
        t = q.get("text", "")
        return t.strip() if isinstance(t, str) else ""
    if isinstance(q, str):
        return q.strip()
    return ""


def _document_tokens(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc = example.get("document", {})
    if not isinstance(doc, dict):
        return []
    toks = doc.get("tokens", {})
    if not isinstance(toks, dict):
        return []

    # NQ stores tokens as parallel arrays: token, start_byte, end_byte, is_html
    token_list = toks.get("token", [])
    start_byte_list = toks.get("start_byte", [])
    end_byte_list = toks.get("end_byte", [])
    is_html_list = toks.get("is_html", [])

    if not all(isinstance(lst, list) for lst in [token_list, start_byte_list, end_byte_list, is_html_list]):
        return []

    min_len = min(len(token_list), len(start_byte_list), len(end_byte_list), len(is_html_list))
    if min_len == 0:
        return []

    # Convert to list of dicts
    result = []
    for i in range(min_len):
        result.append({
            "token": token_list[i],
            "start_byte": start_byte_list[i],
            "end_byte": end_byte_list[i],
            "is_html": is_html_list[i],
        })

    return result


def _render_context_from_tokens(
    toks: List[Dict[str, Any]],
    span: Optional[Tuple[int, int]] = None,
    max_tokens: int = 512,
) -> str:
    """
    Build plain-text context from document.tokens, filtering HTML tokens.
    If span=(start_token, end_token) is provided, restrict to that token range.
    """
    if not toks:
        return ""

    start_i, end_i = 0, len(toks)
    if span is not None:
        s, e = span
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(toks):
            start_i, end_i = s, e

    pieces: List[str] = []
    for tok in toks[start_i:end_i]:
        if not isinstance(tok, dict):
            continue
        if tok.get("is_html", False):
            continue
        t = tok.get("token", "")
        if isinstance(t, str) and t:
            pieces.append(t)
        if len(pieces) >= max_tokens:
            break

    return " ".join(pieces).strip()


def _pick_long_answer_span(example: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    Prefer the first annotation's long_answer span if present.
    """
    anns = example.get("annotations", [])
    if not isinstance(anns, list) or not anns:
        return None

    ann0 = anns[0] if isinstance(anns[0], dict) else None
    if not isinstance(ann0, dict):
        return None

    long_ans = ann0.get("long_answer")
    if isinstance(long_ans, dict):
        s = long_ans.get("start_token")
        e = long_ans.get("end_token")
        if isinstance(s, int) and isinstance(e, int) and s >= 0 and e > s:
            return (s, e)

    long_ans2 = ann0.get("long_answers")
    if isinstance(long_ans2, dict):
        s = long_ans2.get("start_token")
        e = long_ans2.get("end_token")
        if isinstance(s, int) and isinstance(e, int) and s >= 0 and e > s:
            return (s, e)

    return None


def _pick_gold_answers(example: Dict[str, Any]) -> List[str]:
    """
    Official NQ HF schema:
      annotations: dict with keys like 'short_answers', 'yes_no_answer'
        short_answers: list of dicts with "text"
        yes_no_answer: label (NO=0, YES=1), example sometimes uses -1 for none
    """
    answers: List[str] = []

    anns = example.get("annotations", {})
    if isinstance(anns, dict):
        # short answers
        sa = anns.get("short_answers", [])
        if isinstance(sa, list):
            for s in sa:
                if isinstance(s, dict):
                    t = s.get("text", "")
                    # Handle both string and list formats
                    if isinstance(t, str) and t.strip():
                        answers.append(t.strip())
                    elif isinstance(t, list):
                        for text_item in t:
                            if isinstance(text_item, str) and text_item.strip():
                                answers.append(text_item.strip())

        # yes/no answers
        yn = anns.get("yes_no_answer", None)
        # Handle both int label or string
        if isinstance(yn, int):
            if yn == 1:
                answers.append("yes")
            elif yn == 0:
                answers.append("no")
        elif isinstance(yn, str):
            if yn.strip().lower() in {"yes", "no"}:
                answers.append(yn.strip().lower())

    # Dedup while preserving order
    seen = set()
    out: List[str] = []
    for a in answers:
        if a not in seen:
            seen.add(a)
            out.append(a)

    return out


def _has_valid_answer_and_context(example: Dict[str, Any], max_ctx_tokens: int) -> Optional[Dict[str, Any]]:
    q = _safe_get_question_text(example)
    if not q:
        return None

    gold_answers = _pick_gold_answers(example)
    if not gold_answers:
        return None

    toks = _document_tokens(example)
    if not toks:
        return None

    span = _pick_long_answer_span(example)
    ctx = _render_context_from_tokens(toks, span=span, max_tokens=max_ctx_tokens)
    if not ctx:
        ctx = _render_context_from_tokens(toks, span=None, max_tokens=max_ctx_tokens)
    if not ctx:
        return None

    ex_id = example.get("id", "")
    ex_id = str(ex_id) if ex_id is not None else ""
    return {"id": ex_id, "question": q, "gold_answers": gold_answers, "gold_context": ctx}


def _build_unanswerable_context(context_pool: List[str], gold_context: str, rng: random.Random) -> str:
    for _ in range(50):
        cand = rng.choice(context_pool)
        if cand and cand != gold_context:
            return cand
    return ""


def _pair_examples(
    base_items: List[Dict[str, Any]],
    context_pool: List[str],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    For each base item, create:
      - answerable row with gold context
      - unanswerable row with mismatched context sampled from `context_pool`

    Samples negative contexts only from within the same data split to prevent cross-split data leakage.
    """
    rows: List[Dict[str, Any]] = []

    for item in base_items:
        q = item["question"]
        gold_answers = item["gold_answers"]
        gold_ctx = item["gold_context"]
        target_answer = gold_answers[0]

        # Create answerable example with gold context
        prompt_a = build_prompt(q, gold_ctx)
        rows.append(
            {
                "id": f'{item["id"]}_a',
                "question": q,
                "context": gold_ctx,
                "gold_answers": gold_answers,
                "label": "answerable",
                "target_text": target_answer,
                "prompt": prompt_a,
                "source_split": item.get("source_split", ""),
            }
        )

        # Create unanswerable example with mismatched context
        neg_ctx = _build_unanswerable_context(context_pool, gold_ctx, rng)
        prompt_u = build_prompt(q, neg_ctx)
        rows.append(
            {
                "id": f'{item["id"]}_u',
                "question": q,
                "context": neg_ctx,
                "gold_answers": gold_answers,
                "label": "unanswerable",
                "target_text": REFUSAL_TOKEN,
                "prompt": prompt_u,
                "source_split": item.get("source_split", ""),
            }
        )

    rng.shuffle(rows)
    return rows


def build_refusal_splits(
    raw_dir: str,
    out_dir: str,
    sizes: SplitSizes,
    seed: int = 42,
    dataset_name: str = "google-research-datasets/natural_questions",
    dataset_config: str = "default",
    max_ctx_tokens: int = 512,
    use_hf_validation_as_test: bool = True,
) -> None:
    """
    Build train/val/test splits for context-aware QA with refusal supervision.

    Creates paired answerable/unanswerable examples by sampling contexts from
    Wikipedia documents. Answerable examples use evidence-containing contexts;
    unanswerable examples use mismatched contexts lacking required evidence.

    Args:
        raw_dir: Directory containing downloaded HF dataset cache
        out_dir: Output directory for JSONL files
        sizes: Train/val/test split sizes
        seed: Random seed for reproducibility
        dataset_name: HuggingFace dataset identifier
        dataset_config: Dataset configuration
        max_ctx_tokens: Maximum context length in tokens
        use_hf_validation_as_test: Use HF validation as test split

    Outputs:
        train.jsonl, val.jsonl, test.jsonl: Training data
        manifest.json: Dataset statistics and configuration

    Each example contains:
        question, context, gold_answers, label, target_text, prompt
    """
    rng = random.Random(seed)
    outp = ensure_dir(out_dir)

    ds = load_dataset(dataset_name, dataset_config, cache_dir=raw_dir)
    if "train" not in ds or "validation" not in ds:
        raise ValueError("Expected 'train' and 'validation' splits in the HF dataset.")

    # Collect train/val base items from HF train
    train_total_base = sizes.train_size + sizes.val_size
    hf_train: Dataset = ds["train"].shuffle(seed=seed)

    # Oversample to account for filtering out invalid examples (3x multiplier)
    hf_train = hf_train.select(range(min(train_total_base * 3, len(hf_train))))

    base_items_trainval: List[Dict[str, Any]] = []
    for ex in tqdm(hf_train, desc="collect train/val base items"):
        parsed = _has_valid_answer_and_context(ex, max_ctx_tokens=max_ctx_tokens)
        if parsed is None:
            continue
        parsed["source_split"] = "train"
        base_items_trainval.append(parsed)
        if len(base_items_trainval) >= train_total_base:
            break

    if len(base_items_trainval) < train_total_base:
        raise RuntimeError(
            f"Could not collect enough valid train/val base items. "
            f"Needed {train_total_base}, got {len(base_items_trainval)}. "
            f"Try increasing oversample or relaxing filters."
        )

    # Reproducible partition
    train_base = base_items_trainval[: sizes.train_size]
    val_base = base_items_trainval[sizes.train_size : sizes.train_size + sizes.val_size]

    # Within-split negative contexts (prevents data leakage across splits)
    train_pool = [b["gold_context"] for b in train_base]
    val_pool = [b["gold_context"] for b in val_base]

    train_rows = _pair_examples(train_base, context_pool=train_pool, rng=rng)
    val_rows = _pair_examples(val_base, context_pool=val_pool, rng=rng)

    # Collect test base items from HF validatio
    test_rows: List[Dict[str, Any]] = []
    base_items_test: List[Dict[str, Any]] = []

    if sizes.test_size > 0:
        if use_hf_validation_as_test:
            hf_val: Dataset = ds["validation"].shuffle(seed=seed + 1)
            hf_val = hf_val.select(range(min(sizes.test_size * 3, len(hf_val))))

            for ex in tqdm(hf_val, desc="collect test base items (HF validation)"):
                parsed = _has_valid_answer_and_context(ex, max_ctx_tokens=max_ctx_tokens)
                if parsed is None:
                    continue
                parsed["source_split"] = "validation"
                base_items_test.append(parsed)
                if len(base_items_test) >= sizes.test_size:
                    break

            if len(base_items_test) < sizes.test_size:
                raise RuntimeError(
                    f"Could not collect enough valid test base items from HF validation. "
                    f"Needed {sizes.test_size}, got {len(base_items_test)}."
                )

            test_pool = [b["gold_context"] for b in base_items_test]
            test_rows = _pair_examples(base_items_test, context_pool=test_pool, rng=rng)
        else:
            # Carving test data from training split risks data leakage.
            raise NotImplementedError(
                "Test data must be sourced from HF validation split. "
                "Set use_hf_validation_as_test=True."
            )

    # Write outputs
    write_jsonl(outp / "train.jsonl", train_rows)
    write_jsonl(outp / "val.jsonl", val_rows)
    write_jsonl(outp / "test.jsonl", test_rows)

    manifest = {
        "seed": seed,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "max_ctx_tokens": max_ctx_tokens,
        "use_hf_validation_as_test": use_hf_validation_as_test,
        "negative_sampling_policy": {
            "train": "train_pool_only",
            "val": "val_pool_only",
            "test": "hf_validation_pool_only" if use_hf_validation_as_test else "n/a",
        },
        "hf_split_sizes": {"train": len(ds["train"]), "validation": len(ds["validation"])},
        "base_items": {
            "train": len(train_base),
            "val": len(val_base),
            "test": len(base_items_test),
        },
        "row_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "label_distribution": {
            "train": {
                "answerable": sum(r["label"] == "answerable" for r in train_rows),
                "unanswerable": sum(r["label"] == "unanswerable" for r in train_rows),
            },
            "val": {
                "answerable": sum(r["label"] == "answerable" for r in val_rows),
                "unanswerable": sum(r["label"] == "unanswerable" for r in val_rows),
            },
            "test": {
                "answerable": sum(r["label"] == "answerable" for r in test_rows),
                "unanswerable": sum(r["label"] == "unanswerable" for r in test_rows),
            },
        },
    }
    write_json(outp / "manifest.json", manifest)
