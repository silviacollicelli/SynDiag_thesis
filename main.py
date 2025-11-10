import numpy as np
import random
import torch
import tqdm
import wandb
import yaml
from model import model
from validation import EarlyStopping, validate_model
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from dataset import MyDataset, train_transform, val_transform
wandb.login()

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


base_dataset = MyDataset(
    config['data']['clinical_path'],
    config['data']['folder_path']
)

labels = [s[1] for s in base_dataset.samples]
cases = [s[2] for s in base_dataset.samples]

cases = np.array(cases)
labels = np.array(labels)

torch.manual_seed(config['seed'])
random.seed(config['seed'])
np.random.seed(config['seed'])

cv = StratifiedGroupKFold(config['k_fold'], shuffle=True)

class_names = ["benign", "malignant"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(labels)), labels, cases)):
    early_stopping = EarlyStopping(config['early_stopping']['patience'], config['early_stopping']['min_delta'])
    print(f"\n=== Fold {fold + 1} / {config['k_fold']} ===")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

    wandb.init(
        project="newcv_project",
        group="cross_validation_run",
        name=f"fold_{fold+1}"
    )

    # --- Create SEPARATE dataset instances for train and val ---
    train_data = MyDataset(
        config['data']['clinical_path'],
        config['data']['folder_path'],
        transform=train_transform
    )
    val_data = MyDataset(
        config['data']['clinical_path'],
        config['data']['folder_path'],
        transform=val_transform
    )

    # --- Subset each independently using the same indices ---
    train_dataset = Subset(train_data, train_idx)
    val_dataset = Subset(val_data, val_idx)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )

    dense, optimizer, criterion, scheduler = model(device)
    print(f"start training {fold+1}")
    for epoch in range(config['epochs']):
        dense.train()
        running_loss = 0.0
  
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = dense(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss/len(train_loader)        
        wandb.log({
            "train_loss": train_loss
        })
        
        # VALIDATION LOOP
        val_loss = validate_model(dense, val_loader, criterion, device, log_images=False, batch_idx=1, class_names=class_names)
        scheduler.step(val_loss)
        print(f"\tEpoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        early_stopping(val_loss)
        if early_stopping.early_stop:                       
            print("Early stopping triggered.")
            break

    wandb.finish()