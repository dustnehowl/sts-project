import torch
import torch.nn as nn
from transformers import AutoModel

class STSModel(nn.Module):
    def __init__(self, model_name: str, hidden_size: int, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 0~1 사이 유사도
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # [CLS]
        return self.regressor(cls)
