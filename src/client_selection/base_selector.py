# src/client_selection/base_selector.py

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class ClientSelector(ABC):
    """客户端选择器抽象基类"""
    
    def __init__(self, num_clients: int, config: Dict[str, Any] = None):
        """
        Args:
            num_clients: 总客户端数量
            config: 配置字典，包含特定分布的参数
        """
        self.num_clients = num_clients
        self.config = config or {}
        self.selection_history = []  # 记录选择历史
        
    @abstractmethod
    def select(self, num_select: int, round_num: int = 0) -> np.ndarray:
        """
        选择客户端
        
        Args:
            num_select: 需要选择的客户端数量
            round_num: 当前轮次（某些策略可能需要）
            
        Returns:
            selected_indices: 被选中的客户端索引数组
        """
        pass
    
    def record_selection(self, selected_indices: np.ndarray, round_num: int):
        """记录本轮选择的客户端"""
        self.selection_history.append({
            'round': round_num,
            'clients': selected_indices.tolist()
        })
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """计算选择统计信息"""
        if not self.selection_history:
            return {}
        
        # 统计每个客户端被选中的次数
        selection_counts = np.zeros(self.num_clients, dtype=int)
        for record in self.selection_history:
            for client_id in record['clients']:
                selection_counts[client_id] += 1
        
        return {
            'selection_counts': selection_counts,
            'mean': float(np.mean(selection_counts)),
            'std': float(np.std(selection_counts)),
            'min': int(np.min(selection_counts)),
            'max': int(np.max(selection_counts)),
            'total_rounds': len(self.selection_history)
        }
    
    def get_name(self) -> str:
        """返回选择器名称"""
        return self.__class__.__name__.replace('Selector', '')
