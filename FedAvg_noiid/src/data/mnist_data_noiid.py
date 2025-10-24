#!/usr/bin/env python3
"""
Non-IID MNIST数据加载器
改进版：创建真正的Non-IID数据分布以展示不同客户端选择策略的差异
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


class NonIIDMNISTData:
    """Non-IID MNIST数据管理器"""
    
    def __init__(self, data_dir='./data', num_clients=100, alpha=0.5, min_samples=50):
        """
        初始化Non-IID MNIST数据
        
        Args:
            data_dir: 数据目录
            num_clients: 客户端数量
            alpha: Dirichlet分布参数，越小越Non-IID
            min_samples: 每个客户端的最小样本数
        """
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.alpha = alpha
        self.min_samples = min_samples
        
        # 数据变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载数据
        self._load_data()
        self._create_noiid_partition()
    
    def _load_data(self):
        """加载MNIST数据集"""
        print("📥 加载MNIST数据集...")
        
        self.train_dataset = datasets.MNIST(
            self.data_dir, 
            train=True, 
            download=True, 
            transform=self.transform
        )
        
        self.test_dataset = datasets.MNIST(
            self.data_dir, 
            train=False, 
            download=True, 
            transform=self.transform
        )
        
        # 提取标签用于分区
        self.train_labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        
        print(f"   训练集大小: {len(self.train_dataset):,}")
        print(f"   测试集大小: {len(self.test_dataset):,}")
    
    def _create_noiid_partition(self):
        """创建Non-IID数据分区"""
        print(f"🔄 创建Non-IID数据分区 (alpha={self.alpha})...")
        
        # 按类别组织数据索引
        class_indices = {}
        for i, label in enumerate(self.train_labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        # 打乱每个类别的索引
        for label in class_indices:
            np.random.shuffle(class_indices[label])
        
        # 使用Dirichlet分布生成每个客户端的类别分布
        num_classes = len(class_indices)
        client_class_distributions = np.random.dirichlet([self.alpha] * num_classes, self.num_clients)
        
        # 计算每个客户端应该获得的样本数
        total_samples = len(self.train_dataset)
        base_samples_per_client = max(self.min_samples, total_samples // self.num_clients)
        
        # 为每个客户端分配数据
        self.client_indices = {}
        class_counters = {label: 0 for label in class_indices}
        
        for client_id in range(self.num_clients):
            client_indices = []
            client_samples = base_samples_per_client
            
            # 根据Dirichlet分布为该客户端分配各类别的样本数
            client_class_counts = (client_class_distributions[client_id] * client_samples).astype(int)
            
            # 确保至少有一些样本
            if client_class_counts.sum() == 0:
                client_class_counts[np.argmax(client_class_distributions[client_id])] = self.min_samples
            
            # 为每个类别分配样本
            for class_label, count in enumerate(client_class_counts):
                if count > 0 and class_counters[class_label] < len(class_indices[class_label]):
                    available_samples = len(class_indices[class_label]) - class_counters[class_label]
                    actual_count = min(count, available_samples)
                    
                    start_idx = class_counters[class_label]
                    end_idx = start_idx + actual_count
                    
                    client_indices.extend(class_indices[class_label][start_idx:end_idx])
                    class_counters[class_label] = end_idx
            
            # 如果样本不够，从其他类别补充
            while len(client_indices) < self.min_samples:
                for class_label in class_indices:
                    if class_counters[class_label] < len(class_indices[class_label]):
                        client_indices.append(class_indices[class_label][class_counters[class_label]])
                        class_counters[class_label] += 1
                        if len(client_indices) >= self.min_samples:
                            break
            
            self.client_indices[client_id] = client_indices
        
        # 打印分布统计
        self._print_distribution_stats()
    
    def _print_distribution_stats(self):
        """打印数据分布统计信息"""
        print(f"📊 Non-IID数据分布统计:")
        
        # 统计每个客户端的类别分布
        client_class_counts = {}
        total_samples_per_client = []
        
        for client_id in range(min(10, self.num_clients)):  # 只显示前10个客户端
            class_count = {}
            indices = self.client_indices[client_id]
            total_samples_per_client.append(len(indices))
            
            for idx in indices:
                label = self.train_labels[idx]
                class_count[label] = class_count.get(label, 0) + 1
            
            client_class_counts[client_id] = class_count
            
            # 打印前5个客户端的详细分布
            if client_id < 5:
                class_dist = [class_count.get(i, 0) for i in range(10)]
                print(f"   客户端{client_id:2d}: {class_dist} (总计: {len(indices)})")
        
        # 计算Non-IID程度
        avg_samples = np.mean(total_samples_per_client)
        std_samples = np.std(total_samples_per_client)
        
        print(f"   平均样本数: {avg_samples:.1f} ± {std_samples:.1f}")
        
        # 计算类别分布的不均匀性
        all_class_distributions = []
        for client_id in range(self.num_clients):
            indices = self.client_indices[client_id]
            class_count = [0] * 10
            for idx in indices:
                label = self.train_labels[idx]
                class_count[label] += 1
            
            # 归一化为概率分布
            total = sum(class_count)
            if total > 0:
                class_prob = [c / total for c in class_count]
                all_class_distributions.append(class_prob)
        
        # 计算平均KL散度（衡量Non-IID程度）
        uniform_dist = [0.1] * 10  # 均匀分布
        kl_divergences = []
        
        for dist in all_class_distributions:
            kl_div = sum(p * np.log(p / q + 1e-10) if p > 0 else 0 
                        for p, q in zip(dist, uniform_dist))
            kl_divergences.append(kl_div)
        
        avg_kl = np.mean(kl_divergences)
        print(f"   Non-IID程度 (平均KL散度): {avg_kl:.4f} (越大越Non-IID)")
        
        if avg_kl < 0.1:
            print("   🟢 数据分布接近IID")
        elif avg_kl < 0.5:
            print("   🟡 中等程度Non-IID")
        else:
            print("   🔴 高度Non-IID")
    
    def get_client_loader(self, client_id, batch_size=10, shuffle=True):
        """获取指定客户端的数据加载器"""
        if client_id not in self.client_indices:
            raise ValueError(f"客户端{client_id}不存在")
        
        indices = self.client_indices[client_id]
        subset = Subset(self.train_dataset, indices)
        
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def get_test_loader(self, batch_size=1000):
        """获取测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def get_client_data_info(self, client_id):
        """获取客户端数据信息"""
        if client_id not in self.client_indices:
            return None
        
        indices = self.client_indices[client_id]
        class_count = {}
        
        for idx in indices:
            label = self.train_labels[idx]
            class_count[label] = class_count.get(label, 0) + 1
        
        return {
            'total_samples': len(indices),
            'class_distribution': class_count,
            'dominant_classes': sorted(class_count.keys(), key=lambda k: class_count[k], reverse=True)[:3]
        }


def create_noiid_loaders(num_clients=100, alpha=0.5, batch_size=10, data_dir='./data', min_samples=50):
    """
    创建Non-IID数据加载器的便捷函数
    
    Args:
        num_clients: 客户端数量
        alpha: Dirichlet分布参数，越小越Non-IID
        batch_size: 批大小
        data_dir: 数据目录
        min_samples: 每个客户端的最小样本数
    
    Returns:
        tuple: (client_loaders, test_loader, data_manager)
    """
    data_manager = NonIIDMNISTData(
        data_dir=data_dir,
        num_clients=num_clients,
        alpha=alpha,
        min_samples=min_samples
    )
    
    # 创建所有客户端的数据加载器
    client_loaders = {}
    for client_id in range(num_clients):
        client_loaders[client_id] = data_manager.get_client_loader(
            client_id, 
            batch_size=batch_size
        )
    
    test_loader = data_manager.get_test_loader()
    
    return client_loaders, test_loader, data_manager
