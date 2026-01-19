import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from typing import Optional, Callable


def set_seed(seed):
    random.seed(seed)  # Set random seed for Python's random module
    np.random.seed(seed)  # Set random seed for NumPy
    torch.manual_seed(seed)  # Set random seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Set random seed for PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure determinism
    torch.backends.cudnn.benchmark = False

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