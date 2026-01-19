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

def set_seed(seed):
    random.seed(seed)  # Set random seed for Python's random module
    np.random.seed(seed)  # Set random seed for NumPy
    torch.manual_seed(seed)  # Set random seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Set random seed for PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure determinism
    torch.backends.cudnn.benchmark = False
    

def pad_by_repeat(x, target_len):
    bag_size, dim = x.shape
    if bag_size == target_len:
        return x

    # how many extra instances we need
    repeat = target_len - bag_size

    # repeat whole bag as many times as needed
    n_full = repeat // bag_size
    remainder = repeat % bag_size

    pads = []
    if n_full > 0:
        pads.append(x.repeat(n_full, 1))
    if remainder > 0:
        pads.append(x[:remainder])

    pad = torch.cat(pads, dim=0) if pads else None
    return torch.cat([x, pad], dim=0)
    
def transmil_collate(batch):
    X_list = [item["X"] for item in batch]
    Y = torch.tensor([item["Y"] for item in batch])

    max_len = max(x.shape[0] for x in X_list)

    padded_X = []
    for x in X_list:
        padded_X.append(pad_by_repeat(x, max_len))

    X = torch.stack(padded_X, dim=0)  # (B, max_len, 1024)
    return {"X": X, "Y": Y}

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