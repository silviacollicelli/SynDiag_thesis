import numpy as np
import random
import torch
import tqdm
import wandb
import yaml
from model import build_model
from validation import EarlyStopping, validate_model, train_model, train, val
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from dataset import MyDataset, train_transform, val_transform

with open("config.yaml", "r") as file:
    base_cfg = yaml.safe_load(file)

defaults={
    "lr_blocks": 1e-4,
    "lr_ratio": 10,
    "freeze_strategy": "last_block",
    "dropout_rate": 0.2,
    "batch_size": 8,
    "epochs": 5,
    "fold": 0,
    "num_frames": 8
}
wandb.login()
wandb.init(config=defaults)
config = wandb.config

torch.manual_seed(base_cfg['seed'])
random.seed(base_cfg['seed'])
np.random.seed(base_cfg['seed'])

base_dataset = MyDataset(
    base_cfg['data']['clinical_path'],
    base_cfg['data']['folder_path'], 
    base_cfg['data']['frames_folder']
)

print("done base dataset")

labels = [s[1] for s in base_dataset.samples]
cases = [s[2] for s in base_dataset.samples]

cases = np.array(cases)
labels = np.array(labels)

cv = StratifiedGroupKFold(base_cfg["k_folds"], shuffle=True)

early_stop = base_cfg["early_stopping"]["do_it"]
class_names = ["benign", "malignant"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = []
val_data = []

all_train_data = MyDataset(
    base_cfg['data']['clinical_path'],
    base_cfg['data']['folder_path'],
    base_cfg['data']['frames_folder'],
    transform=train_transform
)
all_val_data = MyDataset(
    base_cfg['data']['clinical_path'],
    base_cfg['data']['folder_path'],
    base_cfg['data']['frames_folder'],
    transform=val_transform
)

for _, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(labels)), labels, cases)):
    train_data.append(Subset(all_train_data, train_idx))
    val_data.append(Subset(all_val_data, val_idx))
    #print(len(Subset(all_train_data, train_idx)), len(Subset(all_val_data, val_idx)))

early_stopping = EarlyStopping(base_cfg["early_stopping"]["patience"], base_cfg["early_stopping"]["min_delta"])

print(f"Train samples: {len(train_data[config.fold])}, Val samples: {len(val_data[config.fold])}")
train_loader = DataLoader(
    train_data[config.fold], batch_size=config.batch_size, shuffle=True
)
val_loader = DataLoader(
    val_data[config.fold], batch_size=config.batch_size, shuffle=False
)

model, optimizer, criterion, scheduler = build_model(
    device=device,
    lr_blocks=config.lr_blocks,
    lr_ratio=config.lr_ratio,
    dropout_rate=config.dropout_rate,
    freeze_strategy=config.freeze_strategy
)

for epoch in tqdm.tqdm(range(config.epochs)):
    # TRAINING LOOP       
    train_loss = train(model, device, criterion, optimizer, train_loader)

    # VALIDATION LOOP
    val_loss, acc, stop = val(model, device, criterion, val_loader, epoch, additional_metrics=True)
    scheduler.step(val_loss)
    print(f"\tEpoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    wandb.log({
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_accuracy": acc},
        step=epoch
        )
    
    if stop:
        break

wandb.finish()