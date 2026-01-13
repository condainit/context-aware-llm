from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed across all relevant libraries for reproducible experiments.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        p: Path to directory

    Returns:
        Path object for the directory
    """
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    """
    Append a single JSON object as one line to a JSONL file.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def clean_prediction_text(pred: str | None) -> str:
    """
    Lightweight cleanup so evaluation is closer to "final answer".
    """
    if pred is None:
        return ""
    t = pred.strip()

    for prefix in ["answer:", "final answer:", "final:", "assistant:", "response:"]:
        if t.lower().startswith(prefix):
            t = t[len(prefix) :].strip()

    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1].strip()

    return t


def is_refusal(text: str | None, refusal_token: str = "REFUSE") -> bool:
    """
    Detect whether a model prediction constitutes a refusal to answer.

    Uses robust pattern matching to identify refusal responses, allowing for
    minor formatting variations while maintaining precision.

    Args:
        text: Model prediction text
        refusal_token: Token indicating refusal (default: "REFUSE")

    Returns:
        True if prediction is classified as a refusal

    Detection criteria:
      - Exact match with optional trailing punctuation (., !, ?)
      - Token appears as standalone word (case-insensitive)
    """
    if text is None:
        return False

    t = clean_prediction_text(text).strip().lower()
    tok = refusal_token.strip().lower()

    if t in {tok, f"{tok}.", f"{tok}!", f"{tok}?"}:
        return True

    return re.search(rf"\b{re.escape(tok)}\b", t) is not None


def normalize_answer(s: str) -> str:
    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))
