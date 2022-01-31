import torch
import numpy as np

from math import floor
from torch.utils.data import Dataset


class Flow(Dataset):
    def __init__(self, data, samples, windows, lag, horizon, scaler, bs, is_scaler=False):
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
        return self._scaler

    @property
    def is_scaler(self):
        return self._is_scaler

    @property
    def bs(self):
        return self._bs

    @property
    def nodes(self):
        return self._data.shape[-2]
    
    @property
    def features(self):
        return self._data.shape[-1]
    
    def __getitem__(self, idx):
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
    def __init__(self, adj_path, loger, device):
        self._adj = np.load(adj_path)
        self._device = device
        self._loger = loger

        self._loger.add_info('Generating incidentor to describe and operate correlation of nodes in the graph.', 'INFO')
    
    @property
    def adj(self):
        return torch.from_numpy(self._adj).float().to(self._device)

    @property
    def nodes(self):
        return self._adj.shape[-1]
    
    def normalized(self) -> None:
        A = self._adj
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
        D = np.array(np.sum(A, axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5
        diag = np.reciprocal(np.sqrt(D))
        self._adj = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))

    def cheb_poly_approx(self, k):
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
    def __init__(self, data_path, loger, windows, lag, horizon, train_ratio=0.6, val_ratio=0.3, bs=32, is_scaler=False, device='cpu'):
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
        data = np.load(data_path)
        return data

    def generate_scaler(self, train_data):
        scaler = Scaler(train_data)
        return scaler

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        assert self._val_ratio > 0., "Please define validation set."
        return self._val_data
    
    @property
    def test_data(self):
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
