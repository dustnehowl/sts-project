#!/bin/bash

MODEL_NAME="sts-model"
VERSION=$(date +"%Y-%m-%d_%H-%M")
FILE_PATH="outputs/2025-04-17/10-22-24/sts_model.onnx"
S3_FILENAME="model.onnx"

python send_s3.py \
  --model-name "$MODEL_NAME" \
  --version "$VERSION" \
  --file-path "$FILE_PATH" \
  --s3-filename "$S3_FILENAME"
