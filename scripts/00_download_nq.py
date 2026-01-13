from __future__ import annotations

import argparse
from datasets import load_dataset

from src.utils import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data/raw")
    ap.add_argument(
        "--dataset-name",
        type=str,
        default="google-research-datasets/natural_questions",
        help="Hugging Face dataset ID",
    )
    ap.add_argument("--dataset-config", type=str, default="default")
    args = ap.parse_args()

    out = ensure_dir(args.out_dir)

    _ = load_dataset(args.dataset_name, args.dataset_config, cache_dir=str(out))

    print(f"Downloaded {args.dataset_name}/{args.dataset_config} to cache_dir={out}")


if __name__ == "__main__":
    main()
