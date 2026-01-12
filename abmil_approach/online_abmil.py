from torchmil.models import ABMIL
import torch
from torchmil.datasets import BinaryClassificationDataset
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

features_path = base_cfg['data']['features_path']
labels_path = base_cfg['data']['labels_path']
root_dir = base_cfg['data']['root_dir']
annotations_file = base_cfg['data']['annotations_file']
k_folds = base_cfg['k_folds']
seed=base_cfg['seed']
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
defaults={
    "l_rate": 1e-4,
    "batch_size": 64,
    "numb_frames": 16,
    "epochs": 100,
    "fold": 0, 
    "att_dim": 256,
    "att_act": "relu",
    "early_stop": False,
    "train_feat_ex": False
}

wandb.login()
wandb.init(config=defaults)
config = wandb.config

cv = StratifiedKFold(k_folds, shuffle=True)
dataset = ImageBagDataset(root_dir, annotations_file, transform)
#dataset_off = BinaryClassificationDataset(features_path+f"{config.numb_frames}", labels_path, bag_keys=["X", "Y"], verbose=False, load_at_init=False)
bag_labels = [dataset[i]["Y"].item() for i in range(len(dataset))]
train_data = []
val_data = []
#train_data_off = []
#val_data_off = []

for _, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(bag_labels)), bag_labels)):
    train_data.append(Subset(dataset, train_idx))
    val_data.append(Subset(dataset, val_idx))
#    train_data_off.append(Subset(dataset_off, train_idx))
#    val_data_off.append(Subset(dataset_off, val_idx))

train_dataloader = DataLoader(
    train_data[config.fold], batch_size=2, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_data[config.fold], batch_size=2, shuffle=False, collate_fn=collate_fn
)

#print("dataset created")
#train_dataloader_off = DataLoader(
#    train_data_off[config.fold], batch_size=2, shuffle=True, collate_fn=collate_fn
#)
#val_dataloader_off = DataLoader(
#    val_data_off[config.fold], batch_size=2, shuffle=False, collate_fn=collate_fn
#)

torch.manual_seed(seed)
model_freeze, optimizer_freeze, crit_freeze = build_model(config.train_feat_ex, config.l_rate, device)
#torch.manual_seed(seed)
#model_train , optimizer_train, crit_train = build_model(True, config.l_rate, device)
#torch.manual_seed(seed)
#model_off = ABMIL((1024,), 256, "relu").to(device)

#optimizer_off = torch.optim.Adam(model_off.parameters(), 1e-4)

for epoch in range(config.epochs):     
    train_loss, train_acc = train(model_freeze, device, crit_freeze, optimizer_freeze, train_dataloader)
    val_loss, val_acc, stop = val(model_freeze, device, crit_freeze, val_dataloader, epoch, additional_metrics=True)
    #train_lt, train_at = train(model_train, device, crit_train, optimizer_train, train_dataloader)
    #val_lt, val_at, stop = val(model_train, device, crit_train, val_dataloader, epoch, additional_metrics=False)
    wandb.log({
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "additional metrics/train_accuracy": train_acc},
        step=epoch
        )

    #train_loss_off, train_acc_off = train(model_off, device, crit_freeze, optimizer_off, train_dataloader_off)
    #val_loss_off, val_acc_off, stop = val(model_off, device, crit_freeze, val_dataloader_off, epoch, additional_metrics=False)
    print(f"Epoch {epoch} | Train Loss fr: {train_loss: .4f} | Val Loss fr: {val_loss: .4f} | Val Accuracy: {val_acc: .4f}")
    #print(f"Epoch {epoch+1} | Train Loss fr: {train_lf: .4f} | Train Loss off: {train_loss_off: .4f} | Val Loss fr: {val_lf: .4f} | Val Loss off: {val_loss_off: .4f}")
