# Makefile for Context-Aware Question Answering with Refusal Supervision

# Configuration
EPOCHS := 1
BATCH_SIZE := 1
GRAD_ACCUM := 16
LEARNING_RATE := 1.4e-4
MAX_NEW_TOKENS := 64
TENSORBOARD_PORT := 6006

.PHONY: help setup data eval-base train train-with-tb eval-finetune reproduce-all reproduce-with-monitoring clean

# Default target
help:
	@echo "Context-Aware Question Answering with Refusal Supervision"
	@echo ""
	@echo "Available targets:"
	@echo "  setup         - Create conda environment"
	@echo "  data          - Download and process Natural Questions dataset"
	@echo "  eval-base     - Evaluate base Qwen3-4B-Instruct model"
	@echo "  train         - Fine-tune with LoRA refusal supervision"
	@echo "  train-with-tb - Train with automatic TensorBoard monitoring"
	@echo "  eval-finetune - Evaluate fine-tuned LoRA adapter"
	@echo "  reproduce-all - Run complete pipeline"
	@echo "  reproduce-with-monitoring - Run pipeline with TensorBoard"
	@echo "  clean         - Remove outputs directory"
	@echo ""
	@echo "Configuration (modify variables at top of Makefile):"
	@echo "  EPOCHS=$(EPOCHS), BATCH_SIZE=$(BATCH_SIZE), LR=$(LEARNING_RATE)"
	@echo "  TENSORBOARD_PORT=$(TENSORBOARD_PORT)"
	@echo ""
	@echo "Quick start: make reproduce-all"

# Environment setup
setup:
	conda env create -f env.yml
	@echo "Environment created. Run: conda activate context-aware-llm"

# Data preparation
data:
	python -m scripts.00_download_nq --out-dir data/raw
	python -m scripts.01_built_splits \
		--raw-dir data/raw \
		--out-dir data/processed \
		--train-size 60000 \
		--val-size 2000 \
		--test-size 2000 \
		--seed 42 \
		--use-hf-validation-as-test

# Base model evaluation
eval-base:
	python -m scripts.03_eval_refusal \
		--model-name Qwen/Qwen3-4B-Instruct-2507 \
		--data-path data/processed/test.jsonl \
		--out-dir outputs/base_eval \
		--batch-size 16 \
		--max-new-tokens 64

# Fine-tuning
train:
	TOKENIZERS_PARALLELISM=false accelerate launch \
		--num_processes 2 \
		--mixed_precision bf16 \
		scripts/02_train_refusal_lora.py \
		--model-name Qwen/Qwen3-4B-Instruct-2507 \
		--train-path data/processed/train.jsonl \
		--val-path data/processed/val.jsonl \
		--out-dir outputs/lora_refusal \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--grad-accum $(GRAD_ACCUM) \
		--lr $(LEARNING_RATE) \
		--seed 42 \
		--num-workers 2 \
		--log-every-steps 10 \
		--mixed-precision bf16 \
		--save-every-epochs 1

# Fine-tuning with automatic TensorBoard monitoring
train-with-tb:
	@echo "Starting TensorBoard on port $(TENSORBOARD_PORT)..."
	@tensorboard --logdir outputs/lora_refusal/tb --port $(TENSORBOARD_PORT) &
	@sleep 3
	@echo "TensorBoard running at: http://localhost:$(TENSORBOARD_PORT)"
	$(MAKE) train
	@echo "Training complete! TensorBoard still running for monitoring."

# Fine-tuned evaluation
eval-finetune:
	python scripts/03_eval_refusal.py \
		--model-name outputs/lora_refusal/best \
		--data-path data/processed/test.jsonl \
		--out-dir outputs/lora_eval \
		--batch-size 16 \
		--max-new-tokens 64

# Complete reproduction pipeline
reproduce-all: setup data eval-base train eval-finetune

# Complete reproduction pipeline with TensorBoard monitoring
reproduce-with-monitoring: setup data eval-base train-with-tb eval-finetune

# Clean outputs
clean:
	rm -rf outputs/