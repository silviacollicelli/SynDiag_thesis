import os
import cv2
import pandas as pd
from PIL import Image
import yaml
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def video_to_tensors(video_path, frame_skip=1):

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
            frames.append(frame)
        
        idx += 1

    cap.release()
    return frames

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
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
                    
                    if item_entry.name.endswith('.mp4'):    #if entry is a video file
                        frames = video_to_tensors(item_entry.path, frame_skip)
                        for frame in frames:
                            self.samples.append((frame, self.labels_dict[img_labels[self.case_folders[i]]], self.case_folders[i]))

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
