import torchvision.models as models
from torchmil.models import ABMIL
import torch
import torch.nn as nn
from torchmil.datasets import BinaryClassificationDataset
from data_feat import ImageBagDataset, transform, End2EndABMIL
from torch.utils.data import DataLoader, Subset
from torchmil.data import collate_fn
from train_val import train, val
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold

features_path = "C:\\Users\\utente\\Documents\\UNI\\MAGISTRALE\\tesi\\raw_dataset\\artifacts\\embeddings\\features"
labels_path = "C:\\Users\\utente\\Documents\\UNI\\MAGISTRALE\\tesi\\raw_dataset\\artifacts\\embeddings\\labels"
root_dir = "C:\\Users\\utente\\Documents\\UNI\\MAGISTRALE\\tesi\\raw_dataset\\artifacts\\Dataset"
annotations_file = "C:\\Users\\utente\\Documents\\UNI\\MAGISTRALE\\tesi\\raw_dataset\\artifacts\\clinical_case_metadata.parquet"
k_folds=5
seed=0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold=0

cv = StratifiedKFold(k_folds, shuffle=True)
dataset = ImageBagDataset(root_dir, annotations_file, transform)
dataset_off = BinaryClassificationDataset(features_path+f"{16}", labels_path, bag_keys=["X", "Y"], verbose=False, load_at_init=False)
#for i in range(len(dataset)):
#    print(dataset[i]['X'].shape, dataset_off[i]['X'].shape)
#print(len(dataset))
bag_labels = [dataset[i]["Y"].item() for i in range(len(dataset))]
train_data = []
val_data = []
train_data_off = []
val_data_off = []

for _, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(bag_labels)), bag_labels)):
    train_data.append(Subset(dataset, train_idx))
    val_data.append(Subset(dataset, val_idx))
    train_data_off.append(Subset(dataset_off, train_idx))
    val_data_off.append(Subset(dataset_off, val_idx))

train_dataloader = DataLoader(
    train_data[fold], batch_size=2, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_data[fold], batch_size=2, shuffle=False, collate_fn=collate_fn
)

train_dataloader_off = DataLoader(
    train_data_off[fold], batch_size=2, shuffle=True, collate_fn=collate_fn
)
val_dataloader_off = DataLoader(
    val_data_off[fold], batch_size=2, shuffle=False, collate_fn=collate_fn
)

#print(f"train set: {len(train_data[0])}\tval set: {len(val_data[0])}")

torch.manual_seed(seed)
model_freeze = End2EndABMIL(train_last_block=False).to(device)
torch.manual_seed(seed)
model_train = End2EndABMIL(train_last_block=True).to(device)
#torch.manual_seed(seed)
#model_off = ABMIL((1024,), 256, "relu").to(device)

#optimizer_off = torch.optim.Adam(model_off.parameters(), 1e-4)
criterion = nn.BCEWithLogitsLoss()
optimizer_freeze = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_freeze.parameters()),
    lr=1e-4
)
optimizer_train = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_train.parameters()),
    lr=1e-4
)

for epoch in range(20):     
    train_lf, train_af = train(model_freeze, device, criterion, optimizer_freeze, train_dataloader)
    val_lf, val_af, stop = val(model_freeze, device, criterion, val_dataloader, epoch, additional_metrics=False)
    train_lt, train_at = train(model_train, device, criterion, optimizer_train, train_dataloader)
    val_lt, val_at, stop = val(model_train, device, criterion, val_dataloader, epoch, additional_metrics=False)
    #train_loss_off, train_acc_off = train(model_off, device, criterion, optimizer_off, train_dataloader_off)
    #val_loss_off, val_acc_off, stop = val(model_off, device, criterion, val_dataloader_off, epoch, additional_metrics=False)

    print(f"Epoch {epoch+1} | Train Loss fr: {train_lf: .4f} | Train Loss tr: {train_lt: .4f} | Val Loss fr: {val_lf: .4f} | Val Loss tr: {val_lt: .4f}")
