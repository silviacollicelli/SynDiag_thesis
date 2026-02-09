import torch.nn as nn
import wandb
import torch
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    accuracy_score, 
    recall_score
)

def train(model, device, criterion, optimizer, dataloader):
    model.train()

    sum_loss = 0.0
    sum_correct = 0.0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        pred = (out > 0).float()
        sum_correct += (pred == data.y).sum().item()
        sum_loss += loss.item()
        
    train_loss = sum_loss / len(dataloader)
    train_acc = sum_correct / len(dataloader.dataset)

    return train_loss, train_acc

def val(model, device, criterion, dataloader, epoch, additional_metrics=True, additional_tables=False):
    model.eval()

    sum_loss = 0.0
    Y_pred = []
    Y_true = []
    pos_probs = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.float())
            pred = (out > 0).float()
            pos_probs.append(out)
            Y_pred.append(pred)
            Y_true.append(data.y)
            sum_loss += loss.item()

    sigmoid = nn.Sigmoid()
    pos_probs = sigmoid(torch.cat(pos_probs)).tolist()
    all_probs = [(1-p, p) for p in pos_probs]
    Y_true = torch.cat(Y_true).int().tolist()
    Y_pred = torch.cat(Y_pred).int().tolist()
    val_loss = sum_loss / len(dataloader)
    val_acc = accuracy_score(Y_true, Y_pred)

    if additional_metrics:
        precision = precision_score(Y_true, Y_pred, zero_division=0.0)
        recall = recall_score(Y_true, Y_pred)
        specificity = recall_score(Y_true, Y_pred, pos_label=0)
        auc = roc_auc_score(Y_true, pos_probs)
        f1 = f1_score(Y_true, Y_pred)
        wandb.log({
            "f1_score": f1,
            "additional metrics/precision": precision,
            "additional metrics/recall": recall,
            "additional metrics/specificity": specificity,
            "additional metrics/AUC": auc 
            },
            step=epoch
        )
    if additional_tables:
        wandb.log({
            "conf_mat": wandb.plot.confusion_matrix(
                preds=Y_pred,
                y_true=Y_true,
                class_names=["benign", "malignant"],
                title="Risk classification Confusion Matrix"
                ),
            "roc_curve": wandb.plot.roc_curve(
                Y_true, 
                all_probs
                )
            },
            step=epoch
            )
    
    return val_loss, val_acc


