# src/client_selection/normal_selector.py

import numpy as np
from scipy.stats import norm
from typing import Dict, Any
from .base_selector import ClientSelector


class NormalSelector(ClientSelector):
    """正态分布选择器"""
    
    def __init__(self, num_clients: int, config: Dict[str, Any] = None):
        super().__init__(num_clients, config)
        
        # 配置参数
        self.mu = config.get('mu', 0) if config else 0
        self.sigma = config.get('sigma', 1.0) if config else 1.0
        self.mode = config.get('normal_mode', 'weighted') if config else 'weighted'
        self.static = config.get('static', True) if config else True
        
        # 生成权重
        if self.static:
            self.weights = self._generate_normal_weights()
    
    def _generate_normal_weights(self) -> np.ndarray:
        """生成基于正态分布的权重"""
        
        method = self.config.get('weight_method', 'random_scores') if self.config else 'random_scores'
        
        if method == 'random_scores':
            # 方法1：随机生成分数
            scores = np.random.normal(self.mu, self.sigma, self.num_clients)
        elif method == 'spatial':
            # 方法2：基于客户端ID的空间分布
            center = self.config.get('center', self.num_clients // 2) if self.config else self.num_clients // 2
            client_ids = np.arange(self.num_clients)
            distances = np.abs(client_ids - center)
            scores = norm.pdf(distances, loc=0, scale=self.sigma)
        elif method == 'fixed_positions':
            # 方法3：客户端均匀分布在正态曲线上
            positions = np.linspace(-3*self.sigma, 3*self.sigma, self.num_clients)
            scores = norm.pdf(positions, loc=self.mu, scale=self.sigma)
        else:
            # 默认方法：随机分数
            scores = np.random.normal(self.mu, self.sigma, self.num_clients)
        
        # 转换为正值权重
        scores_shifted = scores - scores.min() + 1e-6
        weights = scores_shifted / scores_shifted.sum()
        return weights
    
    def select(self, num_select: int, round_num: int = 0) -> np.ndarray:
        """基于正态分布选择客户端"""
        
        # 动态模式：每轮重新生成
        if not self.static:
            weights = self._generate_normal_weights()
        else:
            weights = self.weights
        
        if self.mode == 'top_k':
            # Top-K选择：选择权重最高的K个
            top_k_indices = np.argsort(weights)[-num_select:][::-1]
            selected = top_k_indices
        else:
            # 加权随机选择
            selected = np.random.choice(
                self.num_clients,
                size=num_select,
                replace=False,
                p=weights
            )
        
        self.record_selection(selected, round_num)
        return selected
