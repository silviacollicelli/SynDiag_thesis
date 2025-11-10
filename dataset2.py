import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold


def video_to_tensors(video_path, frame_skip=1):
    """
    Extract frames from a video and return them as a list of PyTorch tensors.
    No frames are saved to disk.
    
    Args:
        video_path (str): Path to the video file.
        frame_skip (int): Number of frames to skip between reads (1 = every frame).
        
    Returns:
        List[torch.Tensor]: List of frames as tensors (C, H, W), normalized to [0,1].
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:
            # Convert BGR (OpenCV) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to tensor and normalize to [0, 1]
            #tensor = torch.from_numpy(frame).float() / 255.0
            # Change shape from (H, W, C) to (C, H, W)
            #tensor = tensor.permute(2, 0, 1)
            frames.append(frame)
        
        idx += 1

    cap.release()
    return frames


# Example usage
#frames = video_to_tensors("/home/silvia.collicelli/data/Dataset/1OmjAyXN7qUvFRu6e0RN2/lo6A177Av9yn5IhCFG-EO/lo6A177Av9yn5IhCFG-EO.mp4", frame_skip=1)
#print(f"Extracted {len(frames)} frames, each of shape {frames[0].shape}")

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

class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, frame_skip=1, transform=None):
        self.samples = []
        
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

        self.case_folders = [entry.name for entry in os.scandir(self.img_dir) if entry.is_dir()]

        for i in range(len(self.case_folders)):
            case_path = os.path.join(self.img_dir, self.case_folders[i])

            for entry in os.scandir(case_path):
                item_folder_path = os.path.join(case_path, entry.name)

                for item_entry in os.scandir(item_folder_path):

                    if item_entry.name.startswith(entry.name) and (item_entry.name.endswith(('.jpeg', '.png'))):    #if entry is an image file
                        self.samples.append((item_entry.path, self.labels_dict[img_labels[self.case_folders[i]]], self.case_folders[i]))

                    #if item_entry.is_dir() and item_entry.name!='00000' and item_entry.name!='frames' and not_all_frames:
                    #    idx_des_frames.append(int(item_entry.name))
                    
                    if item_entry.name.endswith('.mp4'):    #if entry is a video file
                        #idx_des_frames=sorted(idx_des_frames)
                        frames = video_to_tensors(item_entry.path, frame_skip)
                        for frame in frames:
                            self.samples.append((frame, self.labels_dict[img_labels[self.case_folders[i]]], self.case_folders[i]))
                    #    for s in range(idx):
                    #        self.img_paths.append(os.path.join(frames_path, entry.name, f"frame_{idx_des_frames[s]:04d}.jpeg"))
                    #        self.label_files.append(self.labels_dict[img_labels[case_folders[i]]])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample, label, _ = self.samples[idx]
        if isinstance(sample, str):  # image path
            image = Image.open(sample).convert("RGB")
        else:
            image = Image.fromarray(sample)
        if self.transform:
            image = self.transform(image)

        return image, label
    

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

#data = MyDataset(config['data']['clinical_path'], config['data']['folder_path'])
#print("done dataset")


#def train_val_subsets(data, train_transform, val_transform):
#
#    random.shuffle(data.case_folders)
#    train_cases = data.case_folders[:int(0.8*len(data.case_folders))]
#    val_cases = data.case_folders[int(0.8*len(data.case_folders)):]
#
#    train_idx = []
#    val_idx = []
#
#    for i, p in enumerate(data.samples):
#        if p[2] in train_cases:
#            train_idx.append(i)
#        else: 
#            val_idx.append(i)
#
#    train_dataset = torch.utils.data.Subset(data, train_idx)
#    val_dataset  = torch.utils.data.Subset(data, val_idx)
#    train_dataset.transform = train_transform
#    val_dataset.transform = val_transform
#
#    return train_dataset, val_dataset

#train_sub, val_sub = train_val_subsets(data)
#print("divided into training and validation per patient")


#add cross validation with stratifiedGroupkfold
#def cross_val(k, data):
#    labels = []
#    items = []
#    cases = []
#
#    cv = StratifiedGroupKFold(k, shuffle=True)
#    for _, p in enumerate(data.samples):
#        items.append(p[0])
#        labels.append(p[1])
#        cases.append(p[2])
#
#    cases = np.array(cases)
#
#    for fold, (train_idx, val_idx) in enumerate(cv.split(items, labels, cases)):
#        print(f"Fold {fold} - Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
#    
#
#cross_val(5, data)