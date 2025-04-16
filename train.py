import hydra
from omegaconf import DictConfig
from models.sts_model import STSModel
from data.dataset import STSDataset
from torch.utils.data import DataLoader
from utils import set_seed

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    # Dataset
    train_ds = STSDataset(cfg.train.train_file, cfg.model.name, cfg.train.max_length)
    val_ds = STSDataset(cfg.train.val_file, cfg.model.name, cfg.train.max_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    # Model
    model = STSModel(
        model_name=cfg.model.name,
        hidden_size=cfg.model.hidden_size,
        dropout=cfg.model.dropout
    ).to(cfg.train.device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)
    loss_fn = nn.MSELoss()

    # Train loop
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(cfg.train.device)
            attention_mask = batch['attention_mask'].to(cfg.train.device)
            labels = batch['label'].to(cfg.train.device)

            preds = model(input_ids, attention_mask).squeeze()
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

        # Evaluation
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

        print(f"[Epoch {epoch+1}] Val Loss: {total_val_loss / len(val_loader):.4f}")

if __name__ == "__main__":
    main()
