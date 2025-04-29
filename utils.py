import numpy as np
import random
import torch

def set_seed(seed=2025):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
