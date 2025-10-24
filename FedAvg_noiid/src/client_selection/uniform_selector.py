# src/client_selection/uniform_selector.py

import numpy as np
from typing import Dict, Any
from .base_selector import ClientSelector


class UniformSelector(ClientSelector):
    """均匀分布选择器 - 基准方法"""
    
    def __init__(self, num_clients: int, config: Dict[str, Any] = None):
        super().__init__(num_clients, config)
        self.weights = np.ones(num_clients) / num_clients
    
    def select(self, num_select: int, round_num: int = 0) -> np.ndarray:
        """均匀随机选择"""
        selected = np.random.choice(
            self.num_clients,
            size=num_select,
            replace=False,
            p=self.weights
        )
        self.record_selection(selected, round_num)
        return selected
