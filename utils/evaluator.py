import torch
from torch import Tensor


def mape(hat_y, y, masked_value=torch.tensor(0.)) -> (str, Tensor):
    masked_val_mtx = torch.ones_like(y) * masked_value
    mask = torch.ne(y, masked_val_mtx)

    zeros = torch.zeros_like(y)
    mape = torch.where(mask, (y - hat_y) / y, zeros)
    mape = torch.mean(torch.abs(mape))
    return 'mape', mape * 100
    
def mae(hat_y, y) -> (str, Tensor):
    mae = torch.mean(torch.abs(y-hat_y))
    return 'mae', mae
    
def rmse(hat_y, y) -> (str, Tensor):
    rmse = torch.sqrt(torch.mean(torch.pow(y - hat_y, 2)))
    return 'rmse', rmse
