"""初始化训练环境"""
import torch
import random
import numpy as np

def seed_init(seed):
    """防止各框架出现不稳定，保证可复现

    Args:
        seed: 随机种子
    
    Return:
        无返回值
    """
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def device_init():
    """初始化运算硬件
    
    Args:
        无传入值
    
    Return:
        返回硬件名称
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
