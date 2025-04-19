# export_onnx.py

import argparse
import torch
from transformers import AutoTokenizer
from models.sts_model import STSModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, default="sts_model.onnx")
    parser.add_argument("--hidden_size", type=int, required=True)   # 추가
    parser.add_argument("--dropout", type=float, required=True)     # 추가
    return parser.parse_args()


def main():
    args = parse_args()

    # 모델 생성 및 로드
    model = STSModel(
        model_name=args.model_ckpt,
        hidden_size=args.hidden_size,
        dropout=args.dropout
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    # 토크나이저 및 더미 입력
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    inputs = tokenizer("이 문장은 예시입니다.", "이 문장도 예시입니다.", return_tensors="pt")

    # ONNX 변환
    torch.onnx.export(
        model,
        args=(inputs["input_ids"], inputs["attention_mask"]),
        f=args.onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["similarity_score"],
        opset_version=14,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "similarity_score": {0: "batch_size"}
        }
    )

    print(f"[INFO] ONNX 모델이 저장되었습니다: {args.onnx_path}")


if __name__ == "__main__":
    main()
