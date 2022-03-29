"""管理数据，包括
流量类、邻接矩阵类、数据生成、数据收集、标准化类
"""
import torch
import numpy as np

from math import floor
from torch.utils.data import Dataset


class Flow(Dataset):
    """将数据处理成便于模型调取的格式

    Attributes:
        _data: 数据集，可以是训练集、验证集还有测试集
        _samples: 采样长度
        _windows: 历史划窗大小
        _lag: 对于某些Encoder-Decoder架构的模型，
        在Decoder部分的输入所需的历史长度数据
        _horizon: 预测长度
        _scaler: 用于标准化数据的类
        _is_scaler: 标记是否对数据标准化
        _bs: batchsize
    """
    def __init__(self, data, samples, windows, lag, horizon, scaler, bs, is_scaler=False):
        """初始化"""
        self._data = data  # with shape (Time spans, Nodes, Features)
        self._samples = samples
        self._windows = windows
        self._lag = lag
        self._horizon = horizon
        self._scaler = scaler
        self._is_scaler = is_scaler
        self._bs = bs

        if self._data.dim() == 2:
            self._data = torch.unsqueeze(self._data, 2)

    @property
    def scaler(self):
        """标准化类
        """
        return self._scaler

    @property
    def is_scaler(self):
        """是否对数据标准化
        """
        return self._is_scaler

    @property
    def bs(self):
        """batchsize
        """
        return self._bs

    @property
    def nodes(self):
        """结点数
        """
        return self._data.shape[-2]
    
    @property
    def features(self):
        """原始特征维度
        """
        return self._data.shape[-1]
    
    def __getitem__(self, idx):
        """单次加载输入，解决一次性加载带来的内存不足问题
        """
        x_begin = idx
        x_end = x_begin + self._windows
        y_begin = x_end - self._lag
        y_end = y_begin + self._lag + self._horizon

        seq_x = self._data[x_begin: x_end]
        seq_y = self._data[y_begin: y_end]

        if self._is_scaler:
            seq_x = self._scaler.transform(seq_x)
        
        return seq_x, seq_y
    
    def __len__(self):
        return self._samples - self._windows - self._horizon + 1


class Incidentor():
    """用于操作邻接矩阵

    Attributes:
        _adj: 邻接矩阵
        _device: 运算硬件
        _loger: 用于日志管理
    """
    def __init__(self, adj_path, loger, device):
        """初始化"""
        self._adj = np.load(adj_path)
        self._device = device
        self._loger = loger

        self._loger.add_info('Generating incidentor to describe and operate correlation of nodes in the graph.', 'INFO')
    
    @property
    def adj(self):
        """作为最终输入的邻接矩阵
        """
        return torch.from_numpy(self._adj).float().to(self._device)

    @property
    def nodes(self):
        """邻接矩阵中的结点数量
        """
        return self._adj.shape[-1]
    
    def normalized(self) -> None:
        """计算归一化之后的拉普拉斯矩阵
        """
        A = self._adj
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
        D = np.array(np.sum(A, axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5
        diag = np.reciprocal(np.sqrt(D))
        self._adj = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))

    def cheb_poly_approx(self, k):
        """计算切比雪夫近似
        """
        self.normalized()

        n = self._adj.shape[-1]
        L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(self._adj))

        if k > 1:
            L_list = [np.copy(L0), np.copy(L1)]
            for i in range(k - 2):
                Ln = np.mat(2 * self._adj * L1 - L0)
                L_list.append(np.copy(Ln))
                L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
                
            self._adj = np.concatenate(L_list, axis=-1)
        elif k == 1:
            self._adj = np.asarray(L0)
        else:
            raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{k}".')


class Generator():
    """对于原始数据进行划分"""
    def __init__(self, data_path, loger, windows, lag, horizon, train_ratio=0.6, val_ratio=0.3, bs=32, is_scaler=False, device='cpu'):
        """初始化数据生成器

        Attributes:
            _loger: 用于日志管理
            _data: 原始数据，为ndarray数据类型
            _train_ratio: 训练集占比，与_val_ratio之和不大于1
            _val_ratio: 验证集占比，与_train_ratio之和不大于1
            _train_data: 训练数据集
            _test_data: 测试数据集
            _val_data: 验证数据集
        """
        self._loger = loger
        self._loger.add_info('Generating train, val and test dataset.' , 'INFO')
        
        self._data = self.load_data(data_path)
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio

        # split samples
        samples = len(self._data)
        train_samples = floor(samples * train_ratio)
        val_samples = floor(samples * val_ratio)
        test_samples = samples - train_samples - val_samples + horizon

        # split data
        train_left_border = 0
        train_right_border = train_samples
        train_data = self._data[train_left_border: train_right_border]
        train_data = torch.from_numpy(train_data).float().to(device)

        if val_ratio > 0.:
            val_left_border = train_right_border - horizon
            val_right_border = val_left_border + val_samples
            
            test_left_border = val_right_border - horizon

            test_data = self._data[test_left_border:]
            test_data = torch.from_numpy(test_data).float().to(device)

            val_data = self._data[val_left_border: val_right_border]
            val_data = torch.from_numpy(val_data).float().to(device)
        else:
            test_left_border = train_right_border - horizon
            test_data = self._data[test_left_border:]
            test_data = torch.from_numpy(test_data).float().to(device)

        # generate scaler
        scaler = self.generate_scaler(train_data)
        
        # generate data
        self._train_data = Flow(train_data, train_samples, windows, lag, horizon, scaler, bs, is_scaler)
        self._test_data = Flow(test_data, test_samples, windows, lag, horizon, scaler,bs, is_scaler)

        if val_ratio > 0.:
            self._val_data = Flow(val_data, val_samples, windows, lag, horizon, scaler, bs, is_scaler)
        
    def load_data(self, data_path):
        """加载原始数据
        """
        data = np.load(data_path)
        return data

    def generate_scaler(self, train_data):
        """生成对数据标准化的类
        """
        scaler = Scaler(train_data)
        return scaler

    @property
    def train_data(self):
        """返回训练集
        """
        return self._train_data

    @property
    def val_data(self):
        """返回验证集
        """
        assert self._val_ratio > 0., "Please define validation set."
        return self._val_data
    
    @property
    def test_data(self):
        """返回测试集
        """
        return self._test_data


class Collector():
    def __init__(self) -> None:
        self._test_Y = []
        self._test_hat_Y = []
        self._test_Adj = []
    
    def add_flow(self, y, hat_y):
        self._test_Y.append(y)
        self._test_hat_Y.append(hat_y)

    def add_adj(self, adj):
        self._test_Adj.append(adj)
    
    def merge_flow(self):
        Y = torch.concat(self._test_Y, dim=0)
        hat_Y = torch.concat(self._test_hat_Y, dim=0)

        Y = Y.numpy()
        hat_Y = hat_Y.numpy()
        return Y, hat_Y

    def merge_adj(self):
        Adj = torch.concat(self._test_Adj, dim=0)
        Adj = Adj.numpy()
        return Adj

    def fetch_model_params(self):
        raise NotImplementedError


class Scaler():
    def __init__(self, dataset):
        self._mean = dataset.mean()
        self._std = dataset.std()
    
    def transform(self, x):
        return (x - self._mean) / self._std
    
    def inverse_transform(self, x):
        return (x * self._std) + self._mean
