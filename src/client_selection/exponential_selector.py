# src/client_selection/exponential_selector.py

import numpy as np
from scipy.stats import expon
from typing import Dict, Any
from .base_selector import ClientSelector


class ExponentialSelector(ClientSelector):
    """指数分布选择器"""
    
    def __init__(self, num_clients: int, config: Dict[str, Any] = None):
        super().__init__(num_clients, config)
        
        # 配置参数
        self.lambda_param = config.get('lambda', 1.0) if config else 1.0
        self.mode = config.get('exp_mode', 'decay') if config else 'decay'
        
        # 生成权重
        self.weights = self._generate_exponential_weights()
    
    def _generate_exponential_weights(self) -> np.ndarray:
        """生成基于指数分布的权重"""
        
        if self.mode == 'decay':
            # 方法1：指数衰减（头部客户端权重高）
            client_ids = np.arange(self.num_clients)
            weights = np.exp(-self.lambda_param * client_ids / self.num_clients)
        
        elif self.mode == 'random':
            # 方法2：从指数分布随机采样
            priorities = expon.rvs(scale=1/self.lambda_param, size=self.num_clients)
            weights = priorities
        
        elif self.mode == 'data_decay':
            # 方法3：按数据量排序后指数衰减
            if self.config and 'client_data_sizes' in self.config:
                data_sizes = self.config['client_data_sizes']
            else:
                # 默认按客户端ID递减
                data_sizes = np.arange(self.num_clients, 0, -1)
            
            sorted_indices = np.argsort(data_sizes)[::-1]  # 从大到小
            decay_weights = np.exp(-self.lambda_param * np.arange(self.num_clients) / self.num_clients)
            weights = np.zeros(self.num_clients)
            weights[sorted_indices] = decay_weights
        
        else:
            # 默认指数衰减
            client_ids = np.arange(self.num_clients)
            weights = np.exp(-self.lambda_param * client_ids / self.num_clients)
        
        # 归一化
        weights = weights / weights.sum()
        return weights
    
    def select(self, num_select: int, round_num: int = 0) -> np.ndarray:
        """基于指数分布选择客户端"""
        
        selected = np.random.choice(
            self.num_clients,
            size=num_select,
            replace=False,
            p=self.weights
        )
        
        self.record_selection(selected, round_num)
        return selected
