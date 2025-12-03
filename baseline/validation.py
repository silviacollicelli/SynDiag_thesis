import torch
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device)
    img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]
    return img_tensor.clamp(0, 1)

def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(2)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        img = denormalize(img)
        table.add_data(wandb.Image((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)

def train_model(model, train_loader, device, optimizer, criterion):
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
    return train_loss 

def validate_model(model, valid_dl, loss_func, device, epoch, log_images=False, batch_idx=0, class_names=None, additional_metrics=False):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    correct, val_loss = 0, 0.0
    all_preds, all_labels, all_prob, all_pos_prob = [], [], [], []
    sensitivity, specificity, all_pos, all_neg = 0.0, 0.0, 0, 0
    with torch.no_grad():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pos_prob = outputs.softmax(dim=1)[:,1]
            val_loss += loss_func(outputs, labels)*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_pos_prob.extend(pos_prob.cpu().numpy())
            all_prob.extend(outputs.softmax(dim=1).cpu().numpy())

            # Log one batch of images to the dashboard, always same batch_idx.
            if i==batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
        
        all_labels = np.array(all_labels, dtype=int)
        all_pos_prob = np.array(all_pos_prob, dtype=float)
        all_prob = np.array(all_prob, dtype=float)

        for i in range(len(all_labels)):
            if all_labels[i] == 1: 
                all_pos += 1
                if all_labels[i] == all_preds[i]:
                    sensitivity += 1
            elif all_labels[i] == 0:
                all_neg += 1
                if all_labels[i] == all_preds[i]:
                    specificity += 1
        
        # Compute global metrics
        val_loss /= len(valid_dl.dataset)
        acc = correct / len(valid_dl.dataset)

        if additional_metrics: 
            sensitivity /= all_pos
            specificity /= all_neg
            auc = roc_auc_score(all_labels, all_pos_prob)
            f1 = f1_score(all_labels, all_preds)
            wandb.log({
                "conf_mat": wandb.plot.confusion_matrix(
                    preds=all_preds,
                    y_true=all_labels,
                    class_names=class_names,
                    title="Risk classification Confusion Matrix"
                ), 
                "roc_curve": wandb.plot.roc_curve(
                    all_labels, 
                    all_prob
                ),
                "sensitivity": sensitivity, 
                "specificity": specificity,
                "AUC": auc, "F1-score": f1},
                step=epoch
                )
            

    return float(val_loss), acc


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0