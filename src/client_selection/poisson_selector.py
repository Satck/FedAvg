# src/client_selection/poisson_selector.py

import numpy as np
from scipy.stats import poisson
from typing import Dict, Any
from .base_selector import ClientSelector


class PoissonSelector(ClientSelector):
    """泊松分布选择器"""
    
    def __init__(self, num_clients: int, config: Dict[str, Any] = None):
        super().__init__(num_clients, config)
        
        # 配置参数
        self.lambda_param = config.get('lambda', 5) if config else 5
        self.mode = config.get('poisson_mode', 'static') if config else 'static'
        
        # 静态模式：预先生成权重
        if self.mode == 'static':
            self.weights = self._generate_poisson_weights()
        # 动态模式：每轮重新生成
        # （不预先计算）
    
    def _generate_poisson_weights(self) -> np.ndarray:
        """生成基于泊松分布的权重"""
        priorities = poisson.rvs(mu=self.lambda_param, size=self.num_clients)
        priorities = np.maximum(priorities, 1)  # 最小值为1
        weights = priorities / priorities.sum()
        return weights
    
    def select(self, num_select: int, round_num: int = 0) -> np.ndarray:
        """基于泊松分布选择客户端"""
        
        if self.mode == 'dynamic':
            # 动态：每轮重新生成权重
            weights = self._generate_poisson_weights()
        else:
            # 静态：使用预生成的权重
            weights = self.weights
        
        selected = np.random.choice(
            self.num_clients,
            size=num_select,
            replace=False,
            p=weights
        )
        
        self.record_selection(selected, round_num)
        return selected
