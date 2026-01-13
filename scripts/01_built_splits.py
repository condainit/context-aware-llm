from __future__ import annotations

import argparse

from src.data_prep import SplitSizes, build_refusal_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, default="data/raw")
    ap.add_argument("--out-dir", type=str, default="data/processed")
    ap.add_argument("--train-size", type=int, default=60000)
    ap.add_argument("--val-size", type=int, default=2000)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--dataset-name",
        type=str,
        default="google-research-datasets/natural_questions",
        help="Hugging Face dataset ID",
    )
    ap.add_argument("--dataset-config", type=str, default="default")

    ap.add_argument(
        "--max-ctx-tokens",
        type=int,
        default=512,
        help="Max number of non-HTML document tokens to include in constructed context.",
    )
    ap.add_argument(
        "--use-hf-validation-as-test",
        action="store_true",
        help="If set, build test split from HF validation (recommended).",
    )

    args = ap.parse_args()

    build_refusal_splits(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        sizes=SplitSizes(args.train_size, args.val_size, args.test_size),
        seed=args.seed,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_ctx_tokens=args.max_ctx_tokens,
        use_hf_validation_as_test=args.use_hf_validation_as_test,
    )
    print(f"Wrote splits to {args.out_dir}")


if __name__ == "__main__":
    main()
