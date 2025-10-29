import os
import cv2
import shutil
import tqdm
import random
import pandas as pd
from PIL import Image
import numpy as np
import wandb
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
from sklearn.metrics import roc_auc_score, f1_score

#set determinism
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

#extract the frames from a video and save them in a folder called frames
def extract_save_frames(video_path, output_dir, idx_des_frames=[], not_all_frames=False):    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)       #elimina la cartella già presente se c'è
    os.makedirs(output_dir, exist_ok=True) 
    video = cv2.VideoCapture(video_path)
    i, id, s = 0, 0 ,0

    while True:
        success, frame = video.read()
        if not success:
            break

        if not_all_frames:
            if id < len(idx_des_frames) and i == idx_des_frames[id]:
                frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpeg")
                cv2.imwrite(frame_path, frame)
                s += 1
                id += 1
        else:
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpeg")
            cv2.imwrite(frame_path, frame)
            s += 1

        i+=1  
    return s

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

#specificity = true negative rate
#sensitivity = true positive rate
#malignant 1 -> positive
#benign 0 -> negative

def validate_model(model, valid_dl, loss_func, device, log_images=False, batch_idx=0, class_names=None):
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
        sensitivity /= all_pos
        specificity /= all_neg
        val_loss /= len(valid_dl.dataset)
        acc = correct / len(valid_dl.dataset)
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
            "val_loss": val_loss,
            "val_accuracy": acc, 
            "sensitivity": sensitivity, 
            "specificity": specificity,
            "AUC": auc, "F1-score": f1
        })

    return val_loss

#model set to densenet121 with last layers to finetune
def model():
    dense = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)
    dense.classifier = nn.Linear(dense.classifier.in_features, 2).to(device)
    for name, param in dense.features.named_parameters():
        if "denseblock4" not in name:
            param.requires_grad = False         #requires_grad=False -> freeze the parameters
    
    for module in dense.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    optimizer = torch.optim.Adam([
        {'params': dense.features.denseblock4.parameters(), 'lr': 1e-4},
        {'params': dense.classifier.parameters(), 'lr': 1e-3}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    return dense, optimizer, criterion, scheduler

class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, not_all_frames=True, transform=None, target_transform=None):
        self.img_paths = []
        self.label_files = []
        clinical_table = pd.read_parquet(annotations_file)
        labels_table = clinical_table[["clinical_case", "risk_class"]]
        img_labels = dict(zip(labels_table['clinical_case'], labels_table['risk_class']))
        self.labels_dict = {
            "benign": 0, 
            "malignant": 1,
            "borderline": 1     #merging malignant and borderline
        }
        self.risk_dict = {
            0: "benign", 
            1: "malignant"
        }
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.not_all_frames = not_all_frames

        case_folders = [entry.name for entry in os.scandir(self.img_dir) if entry.is_dir()]

        for i in range(len(case_folders)):
            case_path = os.path.join(self.img_dir, case_folders[i])

            for entry in os.scandir(case_path):
                idx_des_frames = []
                item_folder_path = os.path.join(case_path, entry.name)

                for item_entry in os.scandir(item_folder_path):

                    if item_entry.is_file() and item_entry.name.startswith(entry.name) and (item_entry.name.endswith(('.jpeg', '.png'))):    #if entry is an image file
                        self.img_paths.append(item_entry.path)
                        self.label_files.append(self.labels_dict[img_labels[case_folders[i]]])

                    if item_entry.is_dir() and item_entry.name!='00000' and item_entry.name!='frames' and not_all_frames:
                        idx_des_frames.append(int(item_entry.name))
                
                    if item_entry.is_file() and item_entry.name.endswith('.mp4'):    #if entry is a video file
                        idx = extract_save_frames(item_entry.path, os.path.join(entry.path, "frames"), idx_des_frames=idx_des_frames, not_all_frames=not_all_frames)
                        if not_all_frames==False:
                            for s in range(idx):
                                idx_des_frames.append(s)
                        for s in range(idx):
                            self.img_paths.append(os.path.join(entry.path, "frames", f"frame_{idx_des_frames[s]:04d}.jpeg"))
                            self.label_files.append(self.labels_dict[img_labels[case_folders[i]]])

    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.label_files[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

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

clinical_path = r"C:\Users\utente\Documents\UNI\MAGISTRALE\tesi\naive_baseline\raw_dataset\artifacts\clinical_case_metadata.parquet"
folder_path = r"C:\Users\utente\Documents\UNI\MAGISTRALE\tesi\naive_baseline\raw_dataset\artifacts\Dataset"
full_dataset = MyDataset(clinical_path, folder_path, not_all_frames=True, transform=None)  # raw, no augmentation

# Split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

total_runs=1
epochs=15
class_names = ["benign", "malignant"]
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

wandb.init(
    project="baseline_prova7",
    config={
    "learning_rate": 0.0001,
    "architecture": "DenseNet121",
    "dataset": f"ultrasound subset: {len(full_dataset)} images",
    "epochs": epochs,
    })

for run in range(total_runs):
    #wandb.init(name=f"experiment_{run+1}")
    wandb.config.update({
    "seed": 0,
    "device": str(device),
    "augmentation": "flip+rotation+jitter",
    "optimizer": "Adam",
    "scheduler": "ReduceLROnPlateau"
    })

    dense, optimizer, criterion, scheduler = model()
  
    for epoch in range(epochs):
        dense.train()
        running_loss = 0.0
    
        for images, labels in tqdm.tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = dense(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss/len(train_dataloader)        
        wandb.log({
            "train_loss": train_loss
        })
        
        # VALIDATION LOOP
        val_loss = validate_model(dense, test_dataloader, criterion, device, log_images=True, batch_idx=1, class_names=class_names)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    wandb.finish()


