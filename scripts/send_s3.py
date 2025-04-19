# send_s3.py

import os
import boto3
from dotenv import load_dotenv
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="모델 이름 (예: sts-model)")
    parser.add_argument("--version", type=str, required=True, help="모델 버전 (예: 2025-04-19_15-30)")
    parser.add_argument("--file-path", type=str, required=True, help="업로드할 파일 경로 (예: outputs/sts_model.onnx)")
    parser.add_argument("--s3-filename", type=str, default="model.onnx", help="S3에 저장될 파일 이름")
    return parser.parse_args()

def main():
    load_dotenv()

    # 환경 변수에서 S3 인증 정보 불러오기
    access_key = os.getenv("S3_ACCESS_KEY")
    secret_key = os.getenv("S3_SECRET_KEY")
    bucket_name = os.getenv("S3_BUCKET_NAME")

    args = parse_args()

    # S3 업로드 경로 구성
    s3_key = f"models/{args.model_name}/{args.version}/{args.s3_filename}"

    # S3 클라이언트 생성
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # 파일 업로드
    s3.upload_file(args.file_path, bucket_name, s3_key)

    print(f"[INFO] ✅ {args.file_path} → s3://{bucket_name}/{s3_key} 업로드 완료")

if __name__ == "__main__":
    main()
