import torch
import random
import numpy as np

def seed_init(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def device_init():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
