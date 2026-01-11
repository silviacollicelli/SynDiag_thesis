import os
import cv2
import glob
import pandas as pd
from PIL import Image
import yaml
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def video_to_frames(video_path, output_dir, num_frames=32):
    # Build the expected list of frame paths
    expected_paths = [
        os.path.join(output_dir, f"frame_{i:04d}.jpg")
        for i in range(num_frames)
    ]

    # If directory exists and has all required frames, skip extraction
    if os.path.exists(output_dir):
        existing_frames = glob.glob(os.path.join(output_dir, "frame_*.jpg"))
        if len(existing_frames) == num_frames:
            #print(f"Frames already exist — skipping extraction.")
            return expected_paths

        print(f"Frame count mismatch: found {len(existing_frames)}, expected {num_frames}. Resaving frames...")
    else:
        os.makedirs(output_dir)

    # Extract frames if needed
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        filename = expected_paths[i]
        cv2.imwrite(filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    cap.release()
    print(f"Saved {num_frames} frames to {output_dir}")

    return expected_paths

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.CenterCrop(224),
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
    def __init__(self, 
                 annotations_file, 
                 img_dir, 
                 frames_fold,
                 holsbeke_histo = [['endometrioma', 'cystadenoma-fibroma', 'fibroma'], ['epithelial_invasive']],
                 #holsbeke_histo = [['dermoid', 'serous_cystadenoma'], ['endometrioid_adenocarcinoma', 'high_grade_serous_adenocarcinoma', 'adenocarcinoma', 'clear_cell_carcinoma']],
                 with_frames: bool = True,
                 transform=None):
        self.samples = []
        
        clinical_table = pd.read_parquet(annotations_file)
        img_labels = dict(zip(clinical_table['clinical_case'], clinical_table['holsbeke_histological']))
        #img_labels = dict(zip(clinical_table['clinical_case'], clinical_table['histological']))
        considered_histo = set([h for group in holsbeke_histo for h in group])
        self.histo_dict = {k:v for k, v in img_labels.items() if v in considered_histo}
        self.labels_dict = {
            subtype: i
            for i, group in enumerate(holsbeke_histo)
            for subtype in group
        }
        self.risk_dict = {
            0: "benign", 
            1: "malignant"
        }
        self.img_dir = img_dir
        self.transform = transform

        self.case_folders = [entry.name for entry in os.scandir(self.img_dir) if entry.is_dir()]

        for i in range(len(self.case_folders)):
            if img_labels[self.case_folders[i]] in considered_histo:
                case_path = os.path.join(self.img_dir, self.case_folders[i])
                for entry in os.scandir(case_path):
                    item_folder_path = os.path.join(case_path, entry.name)

                    for item_entry in os.scandir(item_folder_path):

                        if item_entry.name.startswith(entry.name) and (item_entry.name.endswith(('.jpeg', '.png'))):    #if entry is an image file
                            self.samples.append((item_entry.path, self.labels_dict[img_labels[self.case_folders[i]]], self.case_folders[i]))

                        if item_entry.name.endswith('.mp4') and with_frames:    #if entry is a video file
                            
                            out_fold_frames = os.path.join(frames_fold, self.case_folders[i])
                            frames_paths = video_to_frames(item_entry.path, out_fold_frames)
                            for path in frames_paths:
                                self.samples.append((path, self.labels_dict[img_labels[self.case_folders[i]]], self.case_folders[i]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample, label, _ = self.samples[idx]
        if isinstance(sample, str):  # image path
            image = Image.open(sample).convert("RGB")
        else:
            print(f"sample {sample} not found")
        if self.transform:
            image = self.transform(image)

        return image, label
    

with open("config.yaml", "r") as file:
    base_cfg = yaml.safe_load(file)

#data = MyDataset(base_cfg['data']['clinical_path'], base_cfg['data']['folder_path'], base_cfg["data"]["frames_folder"])
#print(len(data))
