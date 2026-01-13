from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


REFUSAL_TOKEN = "REFUSE"


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    train_path: str
    val_path: str
    out_dir: str

    # Optimization
    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    max_length: int = 1024
    seed: int = 42
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Dataloader / runtime
    num_workers: int = 2

    # Accelerate / precision
    mixed_precision: str = "bf16"  # {"no","fp16","bf16"}

    # Logging
    tensorboard: bool = True
    log_every_steps: int = 10
    tb_dir: str = "tb"  # created under out_dir/tb_dir

    # History + checkpoints
    eval_history_file: str = "eval_history.jsonl"
    checkpoints_dir: str = "checkpoints"  # out_dir/checkpoints/
    save_every_epochs: int = 1            # save checkpoint at end of training
    save_every_steps: int = 0             # 0 disables step checkpointing
    resume_from: Optional[str] = None     # path to a checkpoint dir (contains trainer_state.json)

    # ---- Length bucketing (padding reduction) ----
    bucket_by_length: bool = True
    bucket_size: int = 32                 # length // bucket_size groups
    bucket_shuffle: bool = True           # shuffle within buckets + shuffle batch order


@dataclass(frozen=True)
class EvalConfig:
    model_name: str
    data_path: str
    out_dir: str

    # Generation
    max_new_tokens: int = 64
    max_length: int = 1024

    # Throughput
    batch_size: int = 16

    # Repro
    seed: int = 42
