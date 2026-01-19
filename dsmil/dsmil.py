from torchmil.models import DSMIL
from torchmil.datasets import BinaryClassificationDataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torchmil.data import collate_fn
from train_val import train, val
from data import make_deterministic_dataloader, set_seed
import torch.nn as nn
import numpy as np
import wandb
import torch
import yaml

with open("abmil_approach/milconfig.yaml", "r") as file:
    base_cfg = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_path = base_cfg["data"]["features_path"]
labels_path = base_cfg["data"]["labels_path"]
k_folds = base_cfg["k_folds"]
seed = base_cfg['seed']

defaults={
    "l_rate": 1e-4,
    "batch_size": 64,
    "numb_frames": 16,
    "epochs": 10,
    "fold": 0, 
    "att_dim": 256, 
    "nonlq": False,
    "nonlv": 1
}

wandb.login()
wandb.init(config=defaults)
config = wandb.config
set_seed(seed)

dataset = BinaryClassificationDataset(features_path+f"{config.numb_frames}", labels_path, bag_keys=["X", "Y"], verbose=False, load_at_init=False)

cv = StratifiedKFold(k_folds, shuffle=True)
bag_labels = [dataset[i]["Y"].item() for i in range(len(dataset))]

train_data = []
val_data = []

for _, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(bag_labels)), bag_labels)):
    train_data.append(Subset(dataset, train_idx))
    val_data.append(Subset(dataset, val_idx))

train_dataloader = make_deterministic_dataloader(
        dataset=train_data[config.fold],
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        offset=0,
        base_seed=seed,
        sampler=None,
        collate_fn=collate_fn,
    )

val_dataloader = make_deterministic_dataloader(
        dataset=val_data[config.fold],
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        offset=0,
        base_seed=seed,
        sampler=None,
        collate_fn=collate_fn,
    )


in_shape = (dataset[0]["X"].shape[-1],)
criterion = nn.BCEWithLogitsLoss()

if config.nonlv==1:
    non_lin_v=False
    dropout=0.0
elif config.nonlv==0.0:
    non_lin_v=True
    dropout=0.0
elif config.nonlv==0.25:
    non_lin_v=True
    dropout=0.3
elif config.nonlv==0.5:
    non_lin_v=True
    dropout=0.5

model = DSMIL(in_shape, config.att_dim, nonlinear_q=config.nonlq, nonlinear_v=non_lin_v, dropout=dropout)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), config.l_rate)

for epoch in range(config.epochs):      
    train_loss, train_acc = train(model, device, criterion, optimizer, train_dataloader)
    val_loss, val_acc, stop = val(model, device, criterion, val_dataloader, epoch, additional_metrics=True)
    #print(f"\tEpoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")            

    wandb.log({
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "additional metrics/train_accuracy": train_acc},
        step=epoch
        )

    if stop:
        break

wandb.finish()