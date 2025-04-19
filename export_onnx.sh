#!/bin/bash

MODEL_CKPT="google-bert/bert-base-uncased"
OUTPUT_DIR="outputs/2025-04-17/10-22-24"
MODEL_PATH="$OUTPUT_DIR/best.pt"
ONNX_PATH="$OUTPUT_DIR/sts_model.onnx"

# config.yaml에서 사용하는 값과 일치시켜야 함
HIDDEN_SIZE=768
DROPOUT=0.1

python export_onnx.py \
  --model_ckpt "$MODEL_CKPT" \
  --model_path "$MODEL_PATH" \
  --onnx_path "$ONNX_PATH" \
  --hidden_size "$HIDDEN_SIZE" \
  --dropout "$DROPOUT"
