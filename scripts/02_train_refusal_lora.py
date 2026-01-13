from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TrainConfig
from src.train import train_lora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--train-path", type=str, required=True)
    ap.add_argument("--val-path", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)

    # Optimization
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)

    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # Runtime
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # Logging
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--log-every-steps", type=int, default=10)
    ap.add_argument("--tb-dir", type=str, default="tb")

    # Checkpoints / resume
    ap.add_argument("--eval-history-file", type=str, default="eval_history.jsonl")
    ap.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    ap.add_argument("--save-every-epochs", type=int, default=1)
    ap.add_argument("--save-every-steps", type=int, default=0)
    ap.add_argument("--resume-from", type=str, default=None)

    # Length bucketing
    ap.add_argument("--bucket-by-length", action="store_true")
    ap.add_argument("--no-bucket-by-length", action="store_true")
    ap.add_argument("--bucket-size", type=int, default=32)
    ap.add_argument("--bucket-shuffle", action="store_true")
    ap.add_argument("--no-bucket-shuffle", action="store_true")

    args = ap.parse_args()

    # Validate mutually-exclusive flags
    if args.tensorboard and args.no_tensorboard:
        raise ValueError("Choose only one of --tensorboard or --no-tensorboard")
    if args.bucket_by_length and args.no_bucket_by_length:
        raise ValueError("Choose only one of --bucket-by-length or --no-bucket-by-length")
    if args.bucket_shuffle and args.no_bucket_shuffle:
        raise ValueError("Choose only one of --bucket-shuffle or --no-bucket-shuffle")
    if args.bucket_size <= 0:
        raise ValueError("--bucket-size must be > 0")

    # Resolve booleans with config defaults as baseline
    tensorboard = True
    if args.no_tensorboard:
        tensorboard = False
    elif args.tensorboard:
        tensorboard = True

    bucket_by_length = True
    if args.no_bucket_by_length:
        bucket_by_length = False
    elif args.bucket_by_length:
        bucket_by_length = True

    bucket_shuffle = True
    if args.no_bucket_shuffle:
        bucket_shuffle = False
    elif args.bucket_shuffle:
        bucket_shuffle = True

    cfg = TrainConfig(
        model_name=args.model_name,
        train_path=args.train_path,
        val_path=args.val_path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        tensorboard=tensorboard,
        log_every_steps=args.log_every_steps,
        tb_dir=args.tb_dir,
        eval_history_file=args.eval_history_file,
        checkpoints_dir=args.checkpoints_dir,
        save_every_epochs=args.save_every_epochs,
        save_every_steps=args.save_every_steps,
        resume_from=args.resume_from,
        # NEW
        bucket_by_length=bucket_by_length,
        bucket_size=args.bucket_size,
        bucket_shuffle=bucket_shuffle,
    )

    best_dir = train_lora(cfg)
    print(f"Training complete. Best checkpoint: {best_dir}")


if __name__ == "__main__":
    main()
