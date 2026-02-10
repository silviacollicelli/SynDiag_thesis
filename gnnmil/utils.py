import random
import torch
import numpy as np

def set_seed(seed):
    random.seed(seed)  # Set random seed for Python's random module
    np.random.seed(seed)  # Set random seed for NumPy
    torch.manual_seed(seed)  # Set random seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Set random seed for PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure determinism
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
