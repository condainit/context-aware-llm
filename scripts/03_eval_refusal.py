from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import EvalConfig
from src.eval import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--data-path", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = EvalConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    metrics = evaluate(cfg)
    print("Done.")
    for k, v in metrics.items():
        if k != "config":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
