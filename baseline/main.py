import numpy as np
import random
import torch
import tqdm
import wandb
import yaml
from model import build_model
from validation import EarlyStopping, validate_model
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from dataset import MyDataset, train_transform, val_transform

with open("config.yaml", "r") as file:
    base_cfg = yaml.safe_load(file)

base_dataset = MyDataset(
    base_cfg['data']['clinical_path'],
    base_cfg['data']['folder_path']
)

print("done base dataset")

labels = [s[1] for s in base_dataset.samples]
cases = [s[2] for s in base_dataset.samples]

cases = np.array(cases)
labels = np.array(labels)

torch.manual_seed(base_cfg['seed'])
random.seed(base_cfg['seed'])
np.random.seed(base_cfg['seed'])

wandb.login()
cv = StratifiedGroupKFold(base_cfg["k_folds"], shuffle=True)

early_stop = base_cfg["early_stopping"]["do_it"]
class_names = ["benign", "malignant"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

defaults={
    "lr_blocks": 1e-4,
    "lr_ratio": 10,
    "freeze_strategy": "last_block",
    "dropout_rate": 0.2,
    "batch_size": 8,
    "epochs": 15 
}
all_fold_histories = []

for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(labels)), labels, cases)):
    history = {"epoch": [], "train_loss": [], "val_loss": [], "accuracy": []}

    wandb.init(config=defaults,
               project="new_test_cv",
               name=f"fold_{fold+1}")
    config = wandb.config
    early_stopping = EarlyStopping(base_cfg["early_stopping"]["patience"], base_cfg["early_stopping"]["min_delta"])
    print(f"\n=== Fold {fold + 1} ===")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

    # --- Create SEPARATE dataset instances for train and val ---
    train_data = MyDataset(
        base_cfg['data']['clinical_path'],
        base_cfg['data']['folder_path'],
        transform=train_transform
    )
    val_data = MyDataset(
        base_cfg['data']['clinical_path'],
        base_cfg['data']['folder_path'],
        transform=val_transform
    )

    # --- Subset each independently using the same indices ---
    train_dataset = Subset(train_data, train_idx)
    val_dataset = Subset(val_data, val_idx)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    model, optimizer, criterion, scheduler = build_model(
        device=device,
        lr_blocks=config.lr_blocks,
        lr_ratio=config.lr_ratio,
        dropout_rate=config.dropout_rate,
        freeze_strategy=config.freeze_strategy
    )

    for epoch in tqdm.tqdm(range(config.epochs)):
        model.train()
        running_loss = 0.0
  
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss/len(train_loader)        
        #wandb.log({
        #   "train_loss": train_loss
        #})
        
        # VALIDATION LOOP
        val_loss, acc= validate_model(model, val_loader, criterion, device, log_images=False, batch_idx=1, class_names=class_names)
        scheduler.step(val_loss)
        #print(f"\tEpoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if early_stop:
            early_stopping(val_loss)
            if early_stopping.early_stop:                       
                print("Early stopping triggered.")
                break

        wandb.log({
            "val_loss": val_loss,
            "train_loss": train_loss,
            "accuracy": acc
        })

        #history["epoch"].append(epoch)
        #history["train_loss"].append(train_loss)
        #history["val_loss"].append(val_loss)
        #history["accuracy"].append(acc)
    
    #print(f"\tfold {fold+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {acc:.4f}")
    #all_fold_histories.append(history)

#mean_val_loss = np.zeros(config.epochs)
#mean_train_loss = np.zeros(config.epochs)
#mean_accuracy = np.zeros(config.epochs)
#
#for epoch in range(config.epochs):
#    for k in range(base_cfg["k_folds"]):
#        mean_val_loss[epoch] += all_fold_histories[k]["val_loss"][epoch]
#        mean_train_loss[epoch] += all_fold_histories[k]["train_loss"][epoch]
#        mean_accuracy[epoch] += all_fold_histories[k]["accuracy"][epoch]
#
#    wandb.log({
#        "val_loss": mean_val_loss[epoch]/base_cfg["k_folds"],
#        "train_loss": mean_train_loss[epoch]/base_cfg["k_folds"],
#        "accuracy": mean_accuracy[epoch]/base_cfg["k_folds"]
#    })


wandb.finish()