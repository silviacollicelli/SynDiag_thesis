import torch
from data_feat import ImageBagDataset, transform, build_model
from torch.utils.data import DataLoader, Subset
from torchmil.data import collate_fn
from train_val import train, val
import random
import wandb
import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold

with open("milconfig.yaml", "r") as file:
    base_cfg = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_path = base_cfg['data']['features_path']
labels_path = base_cfg['data']['labels_path']
root_dir = base_cfg['data']['root_dir']
annotations_file = base_cfg['data']['annotations_file']
k_folds = base_cfg['k_folds']
seed=base_cfg['seed']

defaults={
    "l_rate": 1e-4,
    "batch_size": 64,
    "numb_frames": 16,
    "epochs": 5,
    "fold": 0, 
    "att_dim": 256,
    "att_act": "relu",
    "early_stop": False,
    "train_feat_ex": False
}

wandb.login()
wandb.init(config=defaults)
config = wandb.config

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

cv = StratifiedKFold(k_folds, shuffle=True)
dataset = ImageBagDataset(root_dir, annotations_file, transform)
bag_labels = [dataset[i]["Y"].item() for i in range(len(dataset))]
train_data = []
val_data = []

for _, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(bag_labels)), bag_labels)):
    train_data.append(Subset(dataset, train_idx))
    val_data.append(Subset(dataset, val_idx))

train_dataloader = DataLoader(
    train_data[config.fold], batch_size=2, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_data[config.fold], batch_size=2, shuffle=False, collate_fn=collate_fn
)

torch.manual_seed(seed)
model, optimizer, criterion = build_model(config.train_feat_ex, config.l_rate, device)

for epoch in range(config.epochs):     
    train_loss, train_acc = train(model, device, criterion, optimizer, train_dataloader)
    val_loss, val_acc, stop = val(model, device, criterion, val_dataloader, epoch, additional_metrics=True)

    wandb.log({
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "additional metrics/train_accuracy": train_acc},
        step=epoch
        )

    print(f"Epoch {epoch} | Train Loss fr: {train_loss: .4f} | Val Loss fr: {val_loss: .4f}")
