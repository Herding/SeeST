"""训练器，完成模型与数据结合
"""

import torch

from torch.utils.data import DataLoader
from utils.supervisor import Collector


class Predictor():
    """训练器，将模型作为输入，并可以训练、验证、预测、保存、加载
    模型
    
    Attributes:
        _loger: 用于日志操作的
        _model: 所输入的模型
        _loss: 训练模型所需的损失函数
        _epochs: 训练次数
        _patience: 在验证集上模型不再更新的最大次数
        _val_res: 保存在验证集上的评价结果（默认是MAE），作为评价模型优劣的依据
        _wait_times: 用于记录模型不再更新的次数
        _metrics: 评价指标，作为一个列表传入
        _collector: 用于表示收集模型的预测结果
        _optim: 训练模型所需的优化器
    """
    
    def __init__(self, optim, model, loss, lr, epochs, patience, metrics,
                 loger, device, is_collectd=True, scheduled_sampling=False) -> None:
        """初始化训练器
        """
        
        self._loger = loger
        self._loger.add_info('Initiating predictor.', 'INFO')

        # for the model
        self._model = model.to(device)
        self._loss = loss.to(device)

        # for fiting & predicting
        self._epochs = epochs
        self._patience = patience

        self._val_res = float('inf')
        self._wait_times = 0

        # for evaluating
        self._metrics = metrics

        # for collecting weights and data
        if is_collectd:
            self._collector = Collector()

        self._scheduled_sampling = scheduled_sampling

        self._optim = self.initiate(optim, self._model, lr)

    def initiate(self, optim, model, lr):
        """结合所输入的模型和学习率，
        初始化优化器
        """
        return optim(model.parameters(), lr=lr)

    def early_stop(self, metrics) -> bool:
        """判断是否早停
        """
        if self._val_res > metrics[0][1]:
            self._wait_times = 0
            self._val_res = metrics[0][1]
        else:
            self._wait_times += 1
        
        if self._wait_times < self._patience:
            return False
        else:
            return True

    def fit(self, train_dataset, val_dataset=None, adj=None) -> None:
        """训练模型
        """
        is_scaler = train_dataset.is_scaler
        scaler = train_dataset.scaler
        train_dataset = DataLoader(train_dataset, batch_size=train_dataset.bs, shuffle=True)
        
        batches_seen = 0

        for epoch in range(self._epochs):
            total_loss = 0.

            for x, y in train_dataset:
                self._optim.zero_grad()

                if adj is not None:
                    hat_y = self._model(x, adj)
                else:
                    if self._scheduled_sampling:
                        hat_y = self._model(x, y, batches_seen)
                    else:
                        hat_y = self._model(x)
                
                if is_scaler:
                    hat_y = scaler.inverse_transform(hat_y)
                
                loss = self._loss(hat_y, y)
                loss.backward()
                self._optim.step()

                total_loss += loss.item()

                batches_seen += 1

            train_loss = total_loss / len(train_dataset)
            
            if val_dataset is not None:
                val_loss, Y, hat_Y = self.predict(val_dataset, adj)
                
                self._loger.add_info(f"Epoch {epoch + 1} | training loss: {train_loss:.6f}, val_loss: {val_loss:6f}", 'INFO')
                # print(f"Epoch {epoch + 1} | training loss: {train_loss}, val_loss: {val_loss}")
                
                metrics = self.evaluate(Y, hat_Y)

                if self.early_stop(metrics):
                    self._loger.add_info(f'Early stop at {epoch + 1}', 'INFO')
                    break
            else:
                self._loger.add_info(f'Epoch {epoch + 1} | training loss: {train_loss:.6f}')
                # print(f"Epoch {epoch + 1} | training loss: {train_loss}")

    def predict(self, dataset, adj=None):
        """预测结果
        """
        is_scaler = dataset.is_scaler
        scaler = dataset.scaler
        dataset = DataLoader(dataset, batch_size=dataset.bs)
        Y = []
        hat_Y = []

        total_loss = 0.
        for x, y in dataset:
            with torch.no_grad():

                if adj is not None:
                    hat_y = self._model(x, adj)
                else:
                    hat_y = self._model(x)
                
                if is_scaler:
                    hat_y = scaler.inverse_transform(hat_y)
                    
                loss = self._loss(hat_y, y)
                total_loss += loss.item()
                Y.append(y)
                hat_Y.append(hat_y)

        return total_loss / len(dataset), torch.cat(Y, dim=0), torch.cat(hat_Y, dim=0)

    def evaluate(self, Y, hat_Y) -> list:
        """评价模型
        """
        horizon = Y.shape[1]

        for h in range(horizon):
            eval_res = [m(hat_Y[:, h, :, :], Y[:, h, :, :]) for m in self._metrics]

            eval_info = f'\nAt next {h + 1} time step: '
            for er in eval_res:
                eval_info += f'{er[0]}: {er[1]:.5f}   '
            self._loger.add_info(eval_info, 'INFO')

        avg_eval = [m(hat_Y, Y) for m in self._metrics]
        eval_info = 'Avg evaluation: '

        for ae in avg_eval:
            eval_info += f'{ae[0]}: {ae[1]:.5f}   '

        self._loger.add_info(eval_info, 'INFO')

        return avg_eval

    def save(self, model_path) -> None:
        """保存模型
        """
        torch.save(self._model, model_path)
        self._loger.add_info('Model saved.', 'INFO')

    def load(self, model_path) -> None:
        """加载模型
        """
        self._model = torch.load(model_path)
        self._loger.add_info('Model loaded.', 'INFO')
