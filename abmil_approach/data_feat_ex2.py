import torch
import cv2
import os
import random
import torch.nn as nn
import numpy as np
from PIL import Image
import imageio as iio
import pandas as pd
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Sampler
from torchmil.models.mil_model import MILModel
from torchmil.nn import AttentionPool, LazyLinear
from torchmil.nn.utils import get_feat_dim
from torchmil.data.collate import pad_tensors
from typing import Literal, Optional, Callable
import torchvision.transforms as T
from tensordict import TensorDict

#@lru_cache(maxsize=None)
#def read_images(image_tuple):
#    """Cache image reading. Takes tuple of image paths/bytes."""
#    return torch.stack([torch.from_numpy(iio.imread(img_src)) for img_src in image_tuple])
#
#def read_embeddings(embedding_tuple):  
#
#    """Cache embedding reading. Takes tuple of embeddings."""
#    return torch.stack([torch.from_numpy(np.array(emb)) for emb in embedding_tuple])


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

transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

class ImageBagDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 annotations_file: str,
                 transform,
                 holsbeke_histo = [['endometrioma', 'cystadenoma-fibroma', 'fibroma'], ['epithelial_invasive']],
                 #holsbeke_histo = [['dermoid', 'serous_cystadenoma'], ['endometrioid_adenocarcinoma', 'high_grade_serous_adenocarcinoma', 'adenocarcinoma', 'clear_cell_carcinoma']],
                 with_frames: bool = True, 
                 numb_frames: int = 16, 
                 ) -> None:
        self.root_dir = root_dir
        self.bags = []
        clinical_table = pd.read_parquet(annotations_file)
        img_labels = dict(zip(clinical_table['clinical_case'], clinical_table['holsbeke_histological']))
        #img_labels = dict(zip(clinical_table['clinical_case'], clinical_table['histological']))
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
                if  with_frames or num_videos<len(sorted(os.listdir(bag_path))):
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

        return TensorDict({
            'X': instances, 
            'Y': torch.tensor(bag_label, dtype=torch.long),
            'bag_idx': idx
        })
    
    def get_bag_names(self) -> list:
        """
        Returns:
            List of bag names.
        """
        return self.bags

    def get_bag_name_from_idx(self, bag_idx: int | torch.Tensor) -> str:
        """
        Get the bag name (string) given its numeric index.

        Arguments:
            bag_idx: Integer index or scalar tensor with the bag index.

        Returns:
            bag_name: Corresponding bag name from `self.bag_names`.
        """
        if isinstance(bag_idx, torch.Tensor):
            bag_idx = int(bag_idx.item())
        return self.bags[bag_idx]

    def decode_bag_names(self, bag_idx_batch: torch.Tensor) -> list[str]:
        """
        Decode a batch of bag indices into their corresponding bag names.

        Arguments:
            bag_idx_batch: 1D tensor of shape (batch_size,) with bag indices.

        Returns:
            List of bag names of length batch_size.
        """
        return [self.bags[int(i)] for i in bag_idx_batch.cpu().tolist()]
    
def set_seed(seed):
    random.seed(seed)  # Set random seed for Python's random module
    np.random.seed(seed)  # Set random seed for NumPy
    torch.manual_seed(seed)  # Set random seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Set random seed for PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure determinism
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    # This will be called *inside* the worker process.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(base_seed, offset=0):
    g = torch.Generator()
    g.manual_seed(base_seed + offset)
    return g
    
#class MILDatasetOnline(Dataset):
    """
    Multiple Instance Learning Dataset for frame-based data.
    Groups frames by item to create bags.
    """

    def __init__(
        self,
        root_dir: str,
        annotations_file: str,
        df: pd.DataFrame,
        data_col:Literal["image", "embeddings"],
        label_col:Literal["risk_class"],
        bag_group: Literal["item", "clinical_case"] = "item",
        transform: Optional[Callable] = None,
        normalization: Optional[Callable] = None
    ):
        """
        Args:
            df: DataFrame with columns ['clinical_case', 'item', 'frame', 'embeddings', 'risk_class']
            embedding_col: Name of the column containing embeddings
            label_col: Name of the column containing bag labels
        """
        self.df = df.copy()
        self.data_col = data_col
        self.label_col = label_col
        self.bag_names = []
        self.transform = transform
        self.normalization = normalization
        self.is_image_data = (data_col == "image")
        self.to_tensor_transform = v2.Lambda(lambda t: t.permute(0, 3, 1, 2))

        if bag_group == "item":
            group_cols = ["clinical_case", "item"]
        elif bag_group == "clinical_case":
            group_cols = ["clinical_case"]
        else:
            raise Exception(f"Bag Group not supported {bag_group}")

        # Group by item to create bags
        self.bags = []

        for bag_id, group in self.df.groupby(group_cols):
            sample_ids = (
                group[["clinical_case", "item", "frame"]]
                .astype(str)
                .agg("/".join, axis=1)
                .to_list()
            )

            # Get bag label (should be same for all frames in the item)
            bag_label = group[label_col].iloc[0]

            self.bag_names.append(str(bag_id))

            # Store indices instead of loading data immediately
            # This allows lazy loading in __getitem__
            bag_indices = group.index.tolist()

            self.bags.append(
                {
                    "sample_id": sample_ids,
                    "indices": bag_indices,  # Store DataFrame indices for lazy loading
                    "num_frames": len(group),
                    "Y": bag_label,
                }
            )

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        bag_indices = bag["indices"]
        
        # Get data as hashable tuple for caching
        data_tuple = tuple(self.df.loc[bag_indices, self.data_col].values)
        
        if self.is_image_data:
            # Cached image reading
            X = read_images(data_tuple)
            X = self.to_tensor_transform(X)
            
            # Apply transforms (not cached)
            if self.transform is not None:
                X = self.transform(X)

            if self.normalization:
                X = self.normalization(X)
                # X = torch.stack([self.normalization(img) for img in X])

        else:
            # Cached embedding reading
            X = read_embeddings(data_tuple)

        return TensorDict({
            "sample_id": bag["sample_id"],
            "X": X,
            "num_frames": bag["num_frames"],
            "Y": bag["Y"],
            "bag_idx": idx,
        })
    

    def get_bag_names(self) -> list:
        """
        Returns:
            List of bag names.
        """
        return self.bag_names

    def get_bag_name_from_idx(self, bag_idx: int | torch.Tensor) -> str:
        """
        Get the bag name (string) given its numeric index.

        Arguments:
            bag_idx: Integer index or scalar tensor with the bag index.

        Returns:
            bag_name: Corresponding bag name from `self.bag_names`.
        """
        if isinstance(bag_idx, torch.Tensor):
            bag_idx = int(bag_idx.item())
        return self.bag_names[bag_idx]

    def decode_bag_names(self, bag_idx_batch: torch.Tensor) -> list[str]:
        """
        Decode a batch of bag indices into their corresponding bag names.

        Arguments:
            bag_idx_batch: 1D tensor of shape (batch_size,) with bag indices.

        Returns:
            List of bag names of length batch_size.
        """
        return [self.bag_names[int(i)] for i in bag_idx_batch.cpu().tolist()]


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
    mini_batch_size: int = 32,
    train_backbone:bool=False
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
    if train_backbone:
        for param in feature_extractor.features.denseblock4.parameters():
            param.requires_grad = True
        feature_extractor.features.denseblock4.train()
        grad_ctx = torch.enable_grad()
    else:
        feature_extractor.eval()
        grad_ctx = torch.no_grad()
    batch_size = data.shape[0]
    masks = masks.to(device=device)
    data = data.to(device=device)
    batch_features = []
    with grad_ctx:
        for batch_idx in range(batch_size):
            # Get the mask for this batch item
            batch_mask = masks[batch_idx]  # Shape: (128,)
            
            # Select only images where mask is 1
            selected_images = data[batch_idx][batch_mask == 1]  # Shape: (num_selected, 3, 224, 224)

            bag_size = selected_images.shape[0]
            bag_features = []
            # Process images in batches
            for start_idx in range(0, bag_size, mini_batch_size):
                end_idx = min(start_idx + batch_size, bag_size)
                mini_batch_images = selected_images[start_idx:end_idx]
                mini_batch_features = feature_extractor(mini_batch_images)
                bag_features.append(mini_batch_features)

            batch_features.append(torch.cat(bag_features, dim=0))

    return pad_tensors(batch_features)

class DenseNet121Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )
        self.model.classifier = nn.Identity()  # returns 1024-dim vector

    def forward(self, x):
        return self.model(x)
    
class ABMIL(MILModel):      #similar to torchmil ABMIL but with trainable feature extractor

    def __init__(
        self,
        device: str,
        in_shape: tuple = None,
        att_dim: int = 128,
        att_act: str = "tanh",
        gated: bool = False,
        feat_ext: Optional[torch.nn.Module] = None,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        train_backbone:bool =False
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            att_act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            gated: If True, use gated attention in the attention pooling.
            feat_ext: Feature extractor.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """
        super().__init__()
        self.device=device
        self.criterion = criterion
        self.train_backbone=train_backbone

        self.feat_ext = feat_ext
        if self.feat_ext is None:
            self.feat_ext = torch.nn.Identity()
            
        if in_shape is not None:
            feat_dim = get_feat_dim(self.feat_ext, in_shape)
        else:
            feat_dim = None
        self.pool = AttentionPool(
            in_dim=feat_dim, att_dim=att_dim, act=att_act, gated=gated
        )

        self.classifier = LazyLinear(in_features=feat_dim, out_features=1)

    def forward(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_att: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """
        if isinstance(self.feat_ext, torch.nn.Identity):
            X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)
        else:
            X, mask = extract_features_batched(
                data=X,
                masks=mask,
                feature_extractor=self.feat_ext,
                device=self.device,
                mini_batch_size=16,
                train_backbone=self.train_backbone
            )

        out_pool = self.pool(X, mask, return_att)  # (batch_size, feat_dim)

        if return_att:
            Z, f = out_pool  # (batch_size, feat_dim), (batch_size, bag_size)
        else:
            Z = out_pool  # (batch_size, feat_dim)

        Y_pred = self.classifier(Z)  # (batch_size, 1)
        Y_pred = Y_pred.squeeze(-1)  # (batch_size,)

        if return_att:
            return Y_pred, f
        else:
            return Y_pred

    def compute_loss(
        self, Y: torch.Tensor, X: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """

        Y_pred = self.forward(X, mask, return_att=False)

        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss}

    def predict(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, mask, return_att=return_inst_pred)