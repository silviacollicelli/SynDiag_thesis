import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from typing import Optional, Callable
import torch.nn as nn

def set_seed(seed):
    random.seed(seed)  # Set random seed for Python's random module
    np.random.seed(seed)  # Set random seed for NumPy
    torch.manual_seed(seed)  # Set random seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Set random seed for PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure determinism
    torch.backends.cudnn.benchmark = False
    
def transmil_collate_bs1(batch):    #collate function for batch_size=1
    # batch is a list of length 1
    item = batch[0]
    X = item["X"].unsqueeze(0)  # (1, bag_size, 1024)
    Y = item["Y"].unsqueeze(0)  # (1,)
    return {"X": X, "Y": Y}

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

def mil_collate_fn(batch):
    """
    batch: list[TensorDict] OR TensorDict
    Each element contains:
      - "X": Tensor(n_i, D)
      - "Y": Tensor or scalar
    """

    # Case 1: PyTorch gives a list of TensorDicts
    if isinstance(batch, list):
        X_list = [b["X"] for b in batch]
        Y = torch.stack([b["Y"] for b in batch])

    # Case 2: PyTorch already gives a TensorDict
    else:
        X_list = batch["X"]
        Y = batch["Y"]

    n_instances = [x.shape[0] for x in X_list]
    N_max = max(n_instances)
    B = len(X_list)
    D = X_list[0].shape[1]

    X_padded = torch.zeros(B, N_max, D)
    mask = torch.zeros(B, N_max, dtype=torch.bool)

    for i, x in enumerate(X_list):
        n = x.shape[0]
        X_padded[i, :n] = x
        mask[i, :n] = True

    return {
        "X": X_padded,
        "mask": mask,
        "Y": Y
    }
    return X_padded, mask, torch.tensor(labels)

class TransformerMILEncoder(nn.Module):
    def __init__(self, d_in, d_model=256, n_heads=4, n_layers=2):
        super().__init__()

        self.embedding = nn.Linear(d_in, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(self, x, mask):
        """
        x: (B, N, D)
        mask: (B, N) — True = real token
        """
        h = self.embedding(x)

        # PyTorch expects: True = ignore
        key_padding_mask = ~mask

        h = self.transformer(
            h,
            src_key_padding_mask=key_padding_mask
        )

        return h
    
class MaskedAttentionPooling(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, h, mask):
        """
        h: (B, N, d)
        mask: (B, N)
        """
        logits = self.attention(h).squeeze(-1)
        logits[~mask] = float("-inf")

        alpha = torch.softmax(logits, dim=1)
        z = torch.sum(alpha.unsqueeze(-1) * h, dim=1)

        return z

class TransformerMIL(nn.Module):
    def __init__(self, d_in, d_model=256, n_heads=4, n_layers=2):
        super().__init__()

        self.encoder = TransformerMILEncoder(
            d_in=d_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )

        self.pooling = MaskedAttentionPooling(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        h = self.encoder(x, mask)
        z = self.pooling(h, mask)
        y_hat = self.classifier(z)
        y_hat = y_hat.float().squeeze(1)
        return y_hat