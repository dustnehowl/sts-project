import os
import hydra
import logging
from hydra.utils import to_absolute_path, get_original_cwd
from omegaconf import DictConfig

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from models.sts_model import STSModel
from data.dataset import STSDataset
from utils import set_seed

# ✅ 로깅 설정: logger 생성
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    # ✅ 절대 경로로 변환
    train_path = to_absolute_path(cfg.train.train_file)
    val_path = to_absolute_path(cfg.train.val_file)

    # ✅ Dataset
    train_ds = STSDataset(train_path, cfg.model.name, cfg.train.max_length)
    val_ds = STSDataset(val_path, cfg.model.name, cfg.train.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    # ✅ Model
    model = STSModel(
        model_name=cfg.model.name,
        hidden_size=cfg.model.hidden_size,
        dropout=cfg.model.dropout
    ).to(cfg.train.device)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')

    # ✅ 프로젝트 루트 기준 저장 경로
    save_root = os.getcwd()

    log.info("🚀 Training started")
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
            input_ids = batch['input_ids'].to(cfg.train.device)
            attention_mask = batch['attention_mask'].to(cfg.train.device)
            labels = batch['label'].to(cfg.train.device)

            preds = model(input_ids, attention_mask).squeeze()
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        log.info(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # ✅ Evaluation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(cfg.train.device)
                attention_mask = batch['attention_mask'].to(cfg.train.device)
                labels = batch['label'].to(cfg.train.device)

                preds = model(input_ids, attention_mask).squeeze()
                loss = loss_fn(preds, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        log.info(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")

        # ✅ Save last.pt
        torch.save(model.state_dict(), os.path.join(save_root, "last.pt"))

        # ✅ Save best.pt
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_root, "best.pt"))
            log.info(f"✅ Best model saved at epoch {epoch+1} (val_loss: {avg_val_loss:.4f})")

    log.info("🏁 Training complete.")


if __name__ == "__main__":
    main()

