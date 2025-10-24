# src/client_selection/base_selector.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class ClientSelector(ABC):
    """
    客户端选择器抽象基类
    
    所有客户端选择策略都应该继承这个基类
    """
    
    def __init__(self, num_clients: int, config: Dict[str, Any]):
        """
        初始化客户端选择器
        
        Args:
            num_clients: 客户端总数
            config: 选择器配置参数
        """
        self.num_clients = num_clients
        self.config = config
        self.selection_history = []
        
    @abstractmethod
    def select(self, num_select: int, round_num: int) -> np.ndarray:
        """
        选择客户端
        
        Args:
            num_select: 需要选择的客户端数量
            round_num: 当前轮次
            
        Returns:
            selected_clients: 选中的客户端ID数组
        """
        pass
    
    def record_selection(self, selected_indices: np.ndarray, round_num: int):
        """
        记录选择历史
        
        Args:
            selected_indices: 选中的客户端ID
            round_num: 轮次
        """
        self.selection_history.append({
            'round': round_num,
            'clients': selected_indices.tolist()
        })
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """
        获取选择统计信息
        
        Returns:
            stats: 统计信息字典
        """
        if not self.selection_history:
            return {}
            
        # 统计每个客户端被选中的次数
        selection_counts = np.zeros(self.num_clients)
        for record in self.selection_history:
            for client_id in record['clients']:
                selection_counts[client_id] += 1
        
        return {
            'selection_counts': selection_counts,
            'mean': np.mean(selection_counts),
            'std': np.std(selection_counts),
            'min': np.min(selection_counts),
            'max': np.max(selection_counts),
            'total_rounds': len(self.selection_history)
        }
    
    def get_name(self) -> str:
        """
        获取选择器名称
        
        Returns:
            name: 选择器名称
        """
        return self.__class__.__name__.replace('Selector', '')
