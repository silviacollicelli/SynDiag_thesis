from torchmil.models import ABMIL
from torchmil.datasets import BinaryClassificationDataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from torchmil.data import collate_fn
from train_val import train, val
import torch.nn as nn
import numpy as np
import random
import wandb
import torch
import yaml

with open("milconfig.yaml", "r") as file:
    base_cfg = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_path = base_cfg["data"]["features_path"]
labels_path = base_cfg["data"]["labels_path"]
k_folds = base_cfg["k_folds"]
seed = base_cfg['seed']

defaults={
    "l_rate": 1e-4,
    "batch_size": 2,
    "numb_frames": 32,
    "epochs": 100,
    "fold": 0, 
    "att_dim": 256,
    "att_act": "relu",
    "early_stop": False,
    "pat_scheduler": 10,
    "factor_scheduler": 0.7,
    "min_lr": 1e-6
}

wandb.login()
wandb.init(config=defaults)
config = wandb.config

dataset = BinaryClassificationDataset(features_path+f"{config.numb_frames}", labels_path, bag_keys=["X", "Y"], verbose=False, load_at_init=False)
print("MIL dataset created")

cv = StratifiedKFold(k_folds, shuffle=True)
bag_labels = [dataset[i]["Y"].item() for i in range(len(dataset))]

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

train_data = []
val_data = []

for _, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(bag_labels)), bag_labels)):
    train_data.append(Subset(dataset, train_idx))
    val_data.append(Subset(dataset, val_idx))

train_dataloader = DataLoader(
    train_data[config.fold], batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_data[config.fold], batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
)

in_shape = (dataset[0]["X"].shape[-1],)
criterion = nn.BCEWithLogitsLoss()
model = ABMIL(in_shape, config.att_dim, att_act=config.att_act, criterion=criterion)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), config.l_rate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, mode='min', patience=config.pat_scheduler, factor=config.factor_scheduler, min_lr=config.min_lr
#)

for epoch in range(config.epochs):
    # TRAINING LOOP       
    train_loss, train_acc = train(model, device, criterion, optimizer, train_dataloader)

    # VALIDATION LOOP
    val_loss, val_acc, stop = val(model, device, criterion, val_dataloader, epoch, additional_metrics=True)
    #scheduler.step(val_loss)
    #print(f"\tEpoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")            

    wandb.log({
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "Additional metrics/train_accuracy": train_acc},
        step=epoch
        )

    if stop:
        break

wandb.finish()