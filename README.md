# 🧐 Korean STS with PyTorch & Hydra

Train and serve a Semantic Textual Similarity (STS) model for Korean using the KLUE STS dataset.This project supports flexible experimentation and team collaboration across model training and deployment.

## 📌 Features

✅ KLUE STS dataset integration

✅ Modular training with Hydra + PyTorch

✅ Save best.pt and last.pt model checkpoints

✅ Logging to both console and train.log


## 🗂 Project Structure
```
sts-project/
├── config/           # Hydra configs (model/train)
├── data/             # Dataset loading (STSDataset)
├── models/           # STSModel (bert + regressor)
├── checkpoints/      # 🔥 Saved models (best.pt, last.pt)
├── train.py          # Main training script
├── utils.py          # Seed setup etc.
```

## 🚀 Getting Started

1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Download and preprocess KLUE STS
```
python data/prepare.py
# Saves data/train.json and data/validation.json
```
3. Train the model
```
python train.py
```
> Trained models will be saved to checkpoints/best.pt and checkpoints/last.ptLogs are saved to outputs/YYYY-MM-DD_HH-MM-SS/train.log

## ⚙️ Configuration with Hydra

Modify training configs in:

- `config/train.yaml` — batch size, learning rate, device, epochs

- `config/model.yaml` — model name, dropout, hidden size

Override configs from CLI:
```
python train.py train.lr=1e-4 train.batch_size=16
```
## 📦 Model Checkpoints

| File                  | Description                        |
|-----------------------|------------------------------------|
| `outputs/YYYY-MM-DD/HH-MM-SS/best.pt` | Best model (lowest val loss)       |
| `outputs/YYYY-MM-DD/HH-MM-SS/last.pt` | Last model from final epoch        |


## ✨ Future Work

[ ] Extendable for API serving (e.g., FastAPI)

## 👥 Authors

🧑‍💼 NLP & Training: @dustehowl, @SeungHo0422

🧑‍🔧 API & Serving: @dustnehowl

## 📄 License

MIT License. See LICENSE for more information.
