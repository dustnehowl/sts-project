import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from hydra.utils import to_absolute_path


class STSDataset(Dataset):
    def __init__(self, file_path, tokenizer_name, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        absolute_path = to_absolute_path(file_path)
        print("🚨 데이터 로딩 경로:", absolute_path)

        with open(absolute_path, 'r') as f:
            self.samples = json.load(f)

        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        inputs = self.tokenizer(
            ex['sentence1'], ex['sentence2'],
            truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['label'] = torch.tensor(ex['score'] / 5.0, dtype=torch.float)  # 정규화
        return inputs
