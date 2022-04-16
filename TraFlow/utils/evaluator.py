"""评价指标"""
import torch
from torch import Tensor


def mape(hat_y, y, masked_value=torch.tensor(0.)):
    """MAPE

    Args:
        hat_y: 预测值
        y: 真实值
        masked_value: 遮掩运算过程中会异常的值，默认为0

    Return:
        ('mape', mape): 评价指标名称，评价结果
    """
    masked_val_mtx = torch.ones_like(y) * masked_value
    mask = torch.ne(y, masked_val_mtx)

    zeros = torch.zeros_like(y)
    mape = torch.where(mask, (y - hat_y) / y, zeros)
    mape = torch.mean(torch.abs(mape))
    return 'mape', mape * 100
    
def mae(hat_y, y):
    """MAE
    
    Args:
        hat_y: 预测值
        y: 真实值

    Return:
        ('mae', mae): 评价指标名称，评价结果
    """
    mae = torch.mean(torch.abs(y-hat_y))
    return 'mae', mae
    
def rmse(hat_y, y):
    """RMSE

    Args:
        hat_y: 预测值
        y: 真实值
    
    Return:
        ('rmse', rmse): 评价指标名称，评价结果
    """
    rmse = torch.sqrt(torch.mean(torch.pow(y - hat_y, 2)))
    return 'rmse', rmse
