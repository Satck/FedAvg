# src/client_selection/binomial_selector.py

import numpy as np
from typing import Dict, Any
from .base_selector import ClientSelector


class BinomialSelector(ClientSelector):
    """二项分布选择器"""
    
    def __init__(self, num_clients: int, config: Dict[str, Any] = None):
        super().__init__(num_clients, config)
        
        # 配置参数
        self.mode = config.get('binomial_mode', 'uniform') if config else 'uniform'
        self.success_prob = config.get('success_prob', 0.5) if config else 0.5
        
        # 生成每个客户端的选择概率
        if self.mode == 'uniform':
            # 所有客户端相同概率
            self.client_probs = np.full(num_clients, self.success_prob)
        elif self.mode == 'heterogeneous':
            # 异构概率 - 使用Beta分布生成
            alpha = config.get('alpha', 2) if config else 2
            beta = config.get('beta', 5) if config else 5
            self.client_probs = np.random.beta(alpha, beta, num_clients)
        elif self.mode == 'data_proportional':
            # 根据客户端数据量分配概率（需要传入数据量信息）
            data_sizes = config.get('client_data_sizes', np.ones(num_clients)) if config else np.ones(num_clients)
            self.client_probs = data_sizes / data_sizes.sum()
        else:
            # 默认使用 Beta 分布
            alpha = config.get('alpha', 2) if config else 2
            beta = config.get('beta', 5) if config else 5
            self.client_probs = np.random.beta(alpha, beta, num_clients)
    
    def select(self, num_select: int, round_num: int = 0) -> np.ndarray:
        """基于二项分布选择客户端"""
        
        # 方法1：每个客户端独立抛硬币
        if self.config and self.config.get('independent_trials', False):
            selected_mask = np.random.binomial(1, self.client_probs)
            selected = np.where(selected_mask == 1)[0]
            
            # 调整数量到目标值
            if len(selected) < num_select:
                # 不足则从未选中的客户端中补充
                remaining = np.where(selected_mask == 0)[0]
                if len(remaining) > 0:
                    additional = np.random.choice(
                        remaining,
                        size=min(num_select - len(selected), len(remaining)),
                        replace=False
                    )
                    selected = np.concatenate([selected, additional])
            elif len(selected) > num_select:
                # 过多则随机剔除
                selected = np.random.choice(selected, size=num_select, replace=False)
        
        # 方法2：基于概率加权采样（推荐）
        else:
            weights = self.client_probs / self.client_probs.sum()
            selected = np.random.choice(
                self.num_clients,
                size=num_select,
                replace=False,
                p=weights
            )
        
        self.record_selection(selected, round_num)
        return selected
