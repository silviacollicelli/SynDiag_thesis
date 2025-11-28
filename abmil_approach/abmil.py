from torchmil.models import ABMIL
from torchmil.datasets import ProcessedMILDataset
from sklearn.model_selection import StratifiedGroupKFold
from torchmil.utils.trainer import Trainer
import numpy as np
import random
import torch
import yaml

with open("milconfig.yaml", "r") as file:
    base_cfg = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_path = base_cfg["data"]["features_path"]
labels_path = base_cfg["data"]["labels_path"]
k_folds = base_cfg["k_folds"]
seed = base_cfg['seed']
early_stop = base_cfg["early_stopping"]["do_it"]
class_names = ["benign", "malignant"]

dataset = ProcessedMILDataset(features_path, labels_path, bag_keys=["X", "Y"])
print("MIL dataset created")

cv = StratifiedGroupKFold(k_folds, shuffle=True)

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


