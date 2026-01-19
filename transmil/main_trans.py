from torchmil.models import TransMIL
from data_trans import transmil_collate, make_deterministic_dataloader, set_seed
from torchmil.datasets import BinaryClassificationDataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from train_val import train, val
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
    "n_layers": 2,
    "n_heads": 4,
    "n_landmarks": None,
    "dropout": 0,
    "use_mlp": False
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
        collate_fn=transmil_collate,
    )

val_dataloader = make_deterministic_dataloader(
        dataset=train_data[config.fold],
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        offset=0,
        base_seed=seed,
        sampler=None,
        collate_fn=transmil_collate,
    )

in_shape = (dataset[0]["X"].shape[-1],)
criterion = nn.BCEWithLogitsLoss()
model = TransMIL(
    in_shape=in_shape, 
    att_dim=config.att_dim, 
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    n_landmarks=config.n_landmarks,
    dropout=config.dropout,
    use_mlp=config.use_mlp
    )
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), config.l_rate)

for epoch in range(config.epochs):      
    train_loss, train_acc = train(model, device, criterion, optimizer, train_dataloader)
    val_loss, val_acc = val(model, device, criterion, val_dataloader, epoch, additional_metrics=True)
    print(f"\tEpoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")            

    wandb.log({
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "additional metrics/train_accuracy": train_acc},
        step=epoch
        )

wandb.finish()