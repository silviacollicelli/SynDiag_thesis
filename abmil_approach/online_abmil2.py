import torch
from data_feat_ex2 import ImageBagDataset, transform, make_deterministic_dataloader, mil_collate_fn, set_seed, ABMIL, DenseNet121Extractor
from torch.utils.data import Subset
from train_val import train, val
import torch.nn as nn
import wandb
import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.multiprocessing as mp

def main():

    #set configuration
    with open("milconfig.yaml", "r") as file:
        base_cfg = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = base_cfg['data']['root_dir']
    annotations_file = base_cfg['data']['annotations_file']
    k_folds = base_cfg['k_folds']
    seed=base_cfg['seed']

    defaults={
        "l_rate": 1e-4,
        "batch_size": 64,
        "numb_frames": 16,
        "epochs": 10,
        "fold": 0, 
        "att_dim": 256,
        "att_act": "relu",
        "early_stop": False,
        "train_feat_ex": True
    }

    wandb.login()
    wandb.init(config=defaults)
    config = wandb.config
    set_seed(seed)

    #create dataset + training and validation sets for each fold
    cv = StratifiedKFold(k_folds, shuffle=True)
    dataset = ImageBagDataset(root_dir, annotations_file, transform)
    bag_labels = [dataset[i]["Y"].item() for i in range(len(dataset))]
    train_data = []
    val_data = []
    print("done dataset")
    for _, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(bag_labels)), bag_labels)):
        train_data.append(Subset(dataset, train_idx))
        val_data.append(Subset(dataset, val_idx))

    #create dataloaders for training and validation sets
    train_loader = make_deterministic_dataloader(
        dataset=train_data[config.fold],
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=False,
        shuffle=True,
        offset=0,
        base_seed=seed,
        sampler=None,
        collate_fn=mil_collate_fn,
    )

    val_loader = make_deterministic_dataloader(
        dataset=val_data[config.fold],
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=False,
        shuffle=False,
        offset=0,
        base_seed=seed,
        sampler=None,
        collate_fn=mil_collate_fn,
    )
    
    #define model = feature extractor + abmil
    in_shape = dataset[0]['X'][0].size()   #size of one single image (3,224,224)
    feature_extractor = DenseNet121Extractor(train_backbone=config.train_feat_ex)

    model = ABMIL(
        device=device,
        in_shape=in_shape,
        att_dim=config.att_dim,
        att_act=config.att_act,
        feat_ext=feature_extractor
        )
    
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), config.l_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    #start training+validation steps
    for epoch in range(config.epochs):     
        train_loss, train_acc = train(model, device, criterion, optimizer, train_loader)
        val_loss, val_acc, _ = val(model, device, criterion, val_loader, epoch, additional_metrics=True)

        wandb.log({
            "val_loss": val_loss,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "additional metrics/train_accuracy": train_acc},
            step=epoch
            )

        print(f"Epoch {epoch} | Train Loss fr: {train_loss: .4f} | Val Loss fr: {val_loss: .4f}")

if __name__ == "__main__":
    mp.freeze_support()   # required on Windows
    main()