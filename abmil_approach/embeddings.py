import os
import cv2
import numpy as np
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import pandas as pd

transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])
def video_to_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)       # Compute indices of frames to extract
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)   # Jump directly to the frame
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR -> RGB
        frames.append(frame)

    cap.release()
    return frames

class ImageBagDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 annotations_file: str,
                 transform,
                 holsbeke_histo = [['endometrioma', 'cystadenoma-fibroma', 'fibroma'], ['epithelial_invasive']],
                 with_frames: bool = False, 
                 numb_frames: int = 16, 
                 ) -> None:
        self.root_dir = root_dir
        self.bags = []
        clinical_table = pd.read_parquet(annotations_file)
        img_labels = dict(zip(clinical_table['clinical_case'], clinical_table['holsbeke_histological']))
        considered_histo = set([h for group in holsbeke_histo for h in group])
        self.histo_dict = {k:v for k, v in img_labels.items() if v in considered_histo}

        for bag in os.scandir(root_dir):
            if bag.name in self.histo_dict.keys():
                num_videos = 0
                bag_path = os.path.join(root_dir, bag.name)
                for item in os.scandir(bag_path):
                    in_item_path = os.path.join(bag_path, item.name)
                    for in_item in os.scandir(in_item_path):
                        if in_item.name.endswith(".mp4"):
                            num_videos+=1
                if with_frames or num_videos<len(sorted(os.listdir(bag_path))):
                    self.bags.append(bag.name)
        
        self.labels_dict = {
            subtype: i
            for i, group in enumerate(holsbeke_histo)
            for subtype in group
        }
        self.risk_dict = {
            0: "benign", 
            1: "malignant"
        }
        self.transform = transform
        self.with_frames = with_frames
        self.numb_frames = numb_frames

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag_name = self.bags[idx]
        bag_path = os.path.join(self.root_dir, bag_name)

        instances = []
        for entry in os.scandir(bag_path):
            item_folder_path = os.path.join(bag_path, entry.name)
            for item_entry in os.scandir(item_folder_path):

                if item_entry.name.startswith(entry.name) and (item_entry.name.endswith(('.jpeg', '.png'))):    #if entry is an image file
                    img = Image.open(os.path.join(bag_path, item_entry)).convert("RGB")
                    instances.append(self.transform(img))

                if item_entry.name.endswith('.mp4') and self.with_frames:    #if entry is a video file
                    frames = video_to_frames(item_entry.path, self.numb_frames)
                    for frame in frames:
                        instances.append(self.transform(frame))

        instances = torch.stack(instances)  # (bag_size, 3, 224, 224)
        bag_label = self.labels_dict[self.histo_dict[bag_name]]

        return instances, bag_name, bag_label

#data = ImageBagDataset("/home/silvia.collicelli/data/Dataset","/home/silvia.collicelli/data/controlled_dataset_metadata.parquet", transform)

inst_bag_loader = DataLoader(
    ImageBagDataset("/home/silvia.collicelli/data/Dataset",
                    "/home/silvia.collicelli/data/controlled_dataset_metadata.parquet", transform),
    batch_size=1,
    shuffle=False
)

class DenseNet121Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )
        self.model.classifier = nn.Identity()  # returns 1024-dim vector

    def forward(self, x):
        return self.model(x)
    
extractor = DenseNet121Extractor().cuda().eval()
out_feat_dir = "/home/silvia.collicelli/project/embeddings/features"
out_labels_dir = "/home/silvia.collicelli/project/embeddings/labels"

def save_bag_features(images_tensor, bag_id, out_feat_dir, out_labels_dir):
    os.makedirs(out_feat_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    with torch.no_grad():
        feats = extractor(images_tensor.cuda())   # (bag_size, 1024)

    feats_np = feats.cpu().numpy()               # convert to numpy

    path = os.path.join(out_feat_dir, f"{bag_id[0]}.npy")
    lab_path = os.path.join(out_labels_dir, f"{bag_id[0]}.npy")
    np.save(path, feats_np)
    np.save(lab_path, bag_label.numpy())

    #print(f"Saved: {path}   shape={feats_np.shape}")


for bag_id, (bag_images, bag_name, bag_label) in enumerate(inst_bag_loader):
    bag_images = bag_images.squeeze(0)
    save_bag_features(bag_images, bag_name, out_feat_dir, out_labels_dir)
