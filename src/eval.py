from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from tqdm import tqdm

from .config import EvalConfig, REFUSAL_TOKEN
from .modeling import generate_text_batch, load_text_model
from .utils import (
    clean_prediction_text,
    ensure_dir,
    is_refusal,
    normalize_answer,
    read_jsonl,
    write_json,
    write_jsonl,
)


def exact_match(pred: str, gold_answers: List[str]) -> float:
    """
    Compute exact match accuracy against ground truth answers.

    Args:
        pred: Model prediction string
        gold_answers: List of acceptable ground truth answers

    Returns:
        1.0 if prediction exactly matches any gold answer, 0.0 otherwise
    """
    p = normalize_answer(clean_prediction_text(pred))
    return 1.0 if any(p == normalize_answer(g) for g in gold_answers) else 0.0


def contains_gold(pred: str, gold_answers: List[str], min_gold_chars: int = 3) -> float:
    """
    Soft correctness metric.

    Returns 1.0 if ANY normalized gold answer appears as a substring of the
    normalized prediction, else 0.0.

    Useful because instruction models often answer in full sentences
    (e.g., "India won ...") rather than emitting only the gold span ("India"),
    which causes strict EM to undercount correctness.

    min_gold_chars helps reduce obvious false positives for very short gold
    answers (e.g., "no", "yes") that can appear incidentally.
    """
    p = normalize_answer(clean_prediction_text(pred))
    if not p:
        return 0.0

    for g in gold_answers:
        gg = normalize_answer(g)
        if not gg:
            continue
        if len(gg) < min_gold_chars:
            continue
        if gg in p:
            return 1.0

    return 0.0


def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    """
    Evaluate a model on context-aware question answering with refusal metrics.

    Computes comprehensive metrics including answer accuracy, refusal rates,
    and failure mode analysis. Saves predictions and diagnostic files.

    Args:
        cfg: Evaluation configuration dataclass

    Returns:
        Dict containing all computed metrics and evaluation metadata

    Metrics computed:
        - EM: Exact match accuracy on answerable questions
        - Contains Gold: Semantic match rate
        - FCAR: False confident answer rate on unanswerable questions
        - CRR: Correct refusal rate on unanswerable questions
        - ARR: Answerable refusal rate
    """
    out_dir = ensure_dir(cfg.out_dir)
    rows = read_jsonl(cfg.data_path)

    bundle = load_text_model(cfg.model_name)

    preds: List[Dict[str, Any]] = []
    false_confident: List[Dict[str, Any]] = []
    over_refusals: List[Dict[str, Any]] = []

    # Metrics accumulators
    em_sum = 0.0
    em_n = 0

    contains_sum = 0.0
    contains_n = 0

    unans_total = 0
    unans_refuse = 0
    unans_answer = 0

    ans_total = 0
    ans_refuse = 0

    bs = max(1, int(cfg.batch_size))

    for i in tqdm(range(0, len(rows), bs), desc="evaluating (batched)"):
        batch_rows = rows[i : i + bs]
        prompts = [r["prompt"] for r in batch_rows]

        raw_preds = generate_text_batch(
            bundle,
            prompts=prompts,
            max_new_tokens=cfg.max_new_tokens,
            max_length=cfg.max_length,
        )

        for r, raw_pred in zip(batch_rows, raw_preds):
            label = r["label"]
            gold_answers = r.get("gold_answers", [])

            pred = clean_prediction_text(raw_pred)
            refused = is_refusal(pred, REFUSAL_TOKEN)

            row_em = None
            row_contains = None

            if label == "answerable":
                ans_total += 1
                if refused:
                    ans_refuse += 1
                    over_refusals.append(
                        {
                            "id": r.get("id"),
                            "question": r.get("question"),
                            "pred": pred,
                            "gold_answers": gold_answers,
                        }
                    )
                else:
                    row_em = exact_match(pred, gold_answers)
                    em_sum += float(row_em)
                    em_n += 1

                    row_contains = contains_gold(pred, gold_answers)
                    contains_sum += float(row_contains)
                    contains_n += 1
            else:
                unans_total += 1
                if refused:
                    unans_refuse += 1
                else:
                    unans_answer += 1
                    false_confident.append(
                        {
                            "id": r.get("id"),
                            "question": r.get("question"),
                            "pred": pred,
                            "gold_answers": gold_answers,
                        }
                    )

            preds.append(
                {
                    "id": r.get("id"),
                    "label": label,
                    "source_split": r.get("source_split", ""),
                    "question": r.get("question"),
                    "pred": pred,
                    "refused": refused,
                    "gold_answers": gold_answers,
                    # Per-row diagnostics
                    "em_if_answered": row_em,
                    "contains_gold_if_answered": row_contains,
                }
            )

    # Core metrics
    em = em_sum / max(em_n, 1)
    contains_rate = contains_sum / max(contains_n, 1)

    fcar = unans_answer / max(unans_total, 1)  # false confident answer rate
    crr = unans_refuse / max(unans_total, 1)   # correct refusal rate
    arr = ans_refuse / max(ans_total, 1)       # answerable refusal rate

    metrics = {
        "em_answerable_nonrefusal": em,
        "contains_gold_answerable_nonrefusal": contains_rate,
        "false_confident_answer_rate_unanswerable": fcar,
        "correct_refusal_rate_unanswerable": crr,
        "answerable_refusal_rate": arr,
        "counts": {
            "answerable_total": ans_total,
            "answerable_refused": ans_refuse,
            "answerable_em_denominator": em_n,
            "answerable_contains_gold_denominator": contains_n,
            "unanswerable_total": unans_total,
            "unanswerable_refused": unans_refuse,
            "unanswerable_answered": unans_answer,
        },
        "failure_mode_counts": {
            "false_confident_examples": len(false_confident),
            "over_refusal_examples": len(over_refusals),
        },
        "config": asdict(cfg),
    }

    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "predictions.jsonl", preds)

    # Diagnostics
    write_jsonl(out_dir / "false_confident_answers.jsonl", false_confident[:500])
    write_jsonl(out_dir / "answerable_refusals.jsonl", over_refusals[:500])

    return metrics
