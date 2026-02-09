from torchmil.datasets import BinaryClassificationDataset
from sklearn.model_selection import StratifiedKFold
from model import GNNsimple, GNNtopk, GNNcluster
from torch_geometric.loader import DataLoader
from dataset import GraphMILDataset
from torch.utils.data import Subset
from train import train, val
from utils import set_seed
import torch.nn as nn
import numpy as np
import wandb
import torch
import yaml

with open("gnnconfig.yaml", "r") as file:
    base_cfg = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_path = base_cfg["data"]["features_path"]
labels_path = base_cfg["data"]["labels_path"]
k_folds = base_cfg["k_folds"]
seed = base_cfg['seed']

defaults = {
    'epochs': 150,
    'l_rate': 1e-4,
    'fold': 0,
    'batch_size': 64,
    'k': 4,                 #[2,3,6]
    'numb_frames': 16,      #better have larger value?
    'hidden_dim': 256,      #[512,256,128]
    'layers': 2,            #[1,2,3]
    'topk_ratio': 0.3,      #[0.1,0.2,0.3]
    'pooling': 'mean',      #[mean, max, attention]
    'model': 'clusters',    #[simple,topk,clusters]
    'C': 1                  #[1,3,5]
}

wandb.login()
wandb.init(config=defaults)
config = wandb.config
set_seed(seed)

mil_dataset = BinaryClassificationDataset(
    features_path+f"{config.numb_frames}", 
    labels_path, 
    bag_keys=["X", "Y"], 
    load_at_init=False, 
    verbose=False
)
graph_dataset = GraphMILDataset(mil_dataset, config.k)

cv = StratifiedKFold(k_folds, shuffle=True)
bag_labels = [graph_dataset[i]["y"].item() for i in range(len(graph_dataset))]
train_data = []
val_data = []

for _, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(bag_labels)), bag_labels)):
    train_data.append(Subset(graph_dataset, train_idx))
    val_data.append(Subset(graph_dataset, val_idx))

train_dataloader = DataLoader(
    train_data[config.fold], 
    batch_size=config.batch_size, 
    shuffle=True
)
val_dataloader = DataLoader(
    val_data[config.fold], 
    batch_size=config.batch_size, 
    shuffle=False
)

in_shape = graph_dataset[0]['x'].size(1)
#combine the different types of model in a single class!
if config.model == 'simple':
    model = GNNsimple(
        in_dim=in_shape, 
        hidden_dim=config.hidden_dim, 
        num_layers=config.layers,
        pooling=config.pooling
    ).to(device)
elif config.model == 'topk':
    model = GNNtopk(
        in_dim=in_shape,
        hidden_dim=config.hidden_dim,
        num_layers=config.layers,
        topk_ratio=config.topk_ratio,
        aggr=config.pooling
    ).to(device)
elif config.model == 'clusters':
    model = GNNcluster(
        in_dim=in_shape,
        hidden_dim=config.hidden_dim,
        num_clusters=config.C
    ).to(device)
else:
    print("Error: you don't have this model!")

optimizer = torch.optim.Adam(model.parameters(), config.l_rate)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(config.epochs):      
    train_loss, train_acc = train(model, device, criterion, optimizer, train_dataloader)
    val_loss, val_acc = val(model, device, criterion, val_dataloader, epoch, additional_metrics=True)
    #print(f"\tEpoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")            

    wandb.log({
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "additional metrics/train_accuracy": train_acc},
        step=epoch
        )

wandb.finish()


