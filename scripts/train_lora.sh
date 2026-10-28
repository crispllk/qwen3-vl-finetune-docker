#!/usr/bin/env bash
set -euo pipefail

# DATASET may be a container-local JSON/JSONL file or an ms-swift dataset ID.
# An optional first positional argument supplies DATASET; remaining arguments
# are appended to swift sft and can override defaults.
dataset="${DATASET:-}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  dataset="$1"
  shift
fi
if [[ -z "$dataset" ]]; then
  echo "Usage: train_lora.sh /data/train.jsonl [swift sft options...]" >&2
  echo "Or set DATASET and optionally MODEL, OUTPUT_DIR, MAX_LENGTH, NPROC_PER_NODE." >&2
  exit 2
fi

python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: use --gpus all and install NVIDIA Container Toolkit on the host"'
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

exec swift sft \
  --model "${MODEL:-Qwen/Qwen3-VL-4B-Instruct}" \
  --model_type qwen3_vl \
  --dataset "$dataset" \
  --tuner_type lora \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --attn_impl sdpa \
  --padding_free false \
  --packing false \
  --freeze_vit true \
  --freeze_aligner true \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --max_length "${MAX_LENGTH:-2048}" \
  --gradient_checkpointing true \
  --dataloader_num_workers 2 \
  --logging_steps 1 \
  --save_steps 100 \
  --save_total_limit 2 \
  --report_to none \
  --output_dir "${OUTPUT_DIR:-/output/qwen3-vl-4b-lora}" \
  "$@"
