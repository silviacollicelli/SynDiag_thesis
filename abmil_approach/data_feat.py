import os
import cv2
import numpy as np
import torch
import yaml
from torchvision import models
from torchmil.data.collate import pad_tensors
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
from PIL import Image
import random
from typing import Optional, Callable
import torch.nn as nn
import pandas as pd
from tensordict import TensorDict
from torchmil.models import ABMIL

transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
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
        frame = Image.fromarray(frame)
        frames.append(frame)

    cap.release()
    return frames
def set_seed(seed):
    random.seed(seed)  # Set random seed for Python's random module
    np.random.seed(seed)  # Set random seed for NumPy
    torch.manual_seed(seed)  # Set random seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Set random seed for PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure determinism
    torch.backends.cudnn.benchmark = False

class ImageBagDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 annotations_file: str,
                 transform,
                 #holsbeke_histo = [['endometrioma', 'cystadenoma-fibroma', 'fibroma'], ['epithelial_invasive']],
                 holsbeke_histo = [['dermoid', 'serous_cystadenoma'], ['endometrioid_adenocarcinoma', 'high_grade_serous_adenocarcinoma', 'adenocarcinoma', 'clear_cell_carcinoma']],
                 with_frames: bool = True, 
                 numb_frames: int = 16, 
                 ) -> None:
        self.root_dir = root_dir
        self.bags = []
        clinical_table = pd.read_parquet(annotations_file)
        #img_labels = dict(zip(clinical_table['clinical_case'], clinical_table['holsbeke_histological']))
        img_labels = dict(zip(clinical_table['clinical_case'], clinical_table['histological']))
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

        return {
            'X': instances, 
            'Y': torch.tensor(bag_label, dtype=torch.long)
        }

def seed_worker(worker_id):
    # This will be called *inside* the worker process.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(base_seed, offset=0):
    g = torch.Generator()
    g.manual_seed(base_seed + offset)
    return g
    
def make_deterministic_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    offset: int = 0,
    base_seed: int = 0,
    sampler: Optional[Sampler] = None,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=make_generator(base_seed, offset=offset),
        persistent_workers=False,
        sampler=sampler,
        collate_fn=collate_fn,
    )

def mil_collate_fn(batch):
    """
    Collate function for MIL batches.
    Pads variable-length bags to the maximum length in the batch and creates a mask.

    Args:
        batch: List of samples from MILDataset (dictionaries)

    Returns:
        Dictionary with batched data including mask
    """
    batch_dict = {}
    key_list = batch[0].keys()

    # Collect all values for each key
    for key in key_list:
        batch_dict[key] = [sample[key] for sample in batch]

    # Process each key
    for key in key_list:
        data_list = batch_dict[key]

        # Check if it's a tensor
        if isinstance(data_list[0], torch.Tensor):
            # If 0-dimensional tensor (scalar), just stack
            if data_list[0].dim() == 0:
                batch_dict[key] = torch.stack(data_list)
            # If multi-dimensional, pad and create mask
            else:
                padded_data, mask = pad_tensors(data_list)
                batch_dict[key] = padded_data
                if "mask" not in batch_dict:
                    batch_dict["mask"] = mask
        # If not a tensor (e.g., item_ids, clinical_cases), keep as list or convert
        else:
            # Try to convert to tensor if possible (e.g., numeric labels)
            try:
                batch_dict[key] = torch.tensor(data_list)
            except (ValueError, TypeError):
                # Keep as list if not convertible
                pass

    return TensorDict(batch_dict)

def extract_features_batched(
    data:torch.Tensor, 
    masks:torch.Tensor,
    feature_extractor: nn.Module,
    device:str,
    mini_batch_size: int = 32
):
    """
    Extract features from a list of variable-length image tensors using batched processing.
    
    Args:
        images_list: List of tensors, each of shape (num_images_i, C, H, W)
        feature_extractor: PyTorch model that extracts features from images
        batch_size: Number of images to process at once
        device: Device to run the model on ('cuda' or 'cpu')
    
    Returns:
        List of feature tensors, one per original bag
        Each tensor has shape (num_images_i, feature_dim)
    """
    batch_size = data.shape[0]
    masks = masks.to(device)
    data = data.to(device)
    batch_features = []

    for batch_idx in range(batch_size):
        batch_mask = masks[batch_idx]
        selected_images = data[batch_idx][batch_mask == 1]

        bag_size = selected_images.shape[0]
        bag_features = []

        for start_idx in range(0, bag_size, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, bag_size)
            mini_batch_images = selected_images[start_idx:end_idx]

            mini_batch_features = feature_extractor(mini_batch_images)
            bag_features.append(mini_batch_features)

        batch_features.append(torch.cat(bag_features, dim=0))

    return pad_tensors(batch_features)

class DenseNet121Backbone(nn.Module):
    def __init__(self, train_last_block=True):
        super().__init__()
        self.model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )
        feat_dim = self.model.classifier.in_features
        self.in_dim_att = (feat_dim,)
        self.model.classifier = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze denseblock4 if requested
        if train_last_block:
            for param in self.model.features.denseblock4.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)  # (N, 1024)

class End2EndABMIL(nn.Module):
    def __init__(self, 
                 att_dim: int = 256,
                 att_act: str = "relu",
                 gated: bool = False,
                 train_last_block: bool = False):
        super().__init__()

        self.backbone = DenseNet121Backbone(train_last_block)
        self.train_last_block = train_last_block
        self.mil = ABMIL(
            in_shape=self.backbone.in_dim_att,
            att_dim=att_dim,
            att_act=att_act,
            gated=gated
        )

    def forward(self, X, mask):
        if isinstance(self.feat_ext, torch.nn.Identity):
            X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)
        else:
            X, mask = extract_features_batched(
                data=X,
                masks=mask,
                feature_extractor=self.feat_ext,
                device=self.device,
                mini_batch_size=8
            )
        
        if not self.train_last_block:
            with torch.no_grad():
                feats = self.backbone(X)
        else:
            feats = self.backbone(X)
            
        #feats = feats.view(B, N, -1)

        return self.mil(feats, mask)
    
    #def forward(self, X, mask):
    #    B, N = X.shape[:2]
    #    X = X.view(B * N, *X.shape[2:])
    #    if not self.train_last_block:
    #        with torch.no_grad():
    #            feats = self.backbone(X)
    #    else:
    #        feats = self.backbone(X)
    #        
    #    feats = feats.view(B, N, -1)
#
    #    return self.mil(feats, mask)
    
def set_frozen_modules_to_eval(model: nn.Module):
    for name, module in model.named_modules():
        # Skip the top-level module itself
        if module is model:
            continue

        params = list(module.parameters(recurse=False))
        if not params:
            continue

        # Check if all parameters in this module are frozen
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
        if all(not p.requires_grad for p in params):
            module.eval()

def build_model(train_last_block,
                lrate_mil,
                device):
                
    model = End2EndABMIL(train_last_block)
    #set_frozen_modules_to_eval(model)
    model.to(device)

    params_to_optimize = []

    params_to_optimize.append({
        'params': model.mil.parameters(),
        'lr': lrate_mil
    })

    if train_last_block:
        backbone_params = []
        for p in model.backbone.parameters():
            if p.requires_grad:
                backbone_params.append(p)
        params_to_optimize.append({
            'params': backbone_params,
            'lr': lrate_mil*0.1
        })

    optimizer = torch.optim.Adam(params_to_optimize)
    criterion = nn.BCEWithLogitsLoss()

    return model, optimizer, criterion

    