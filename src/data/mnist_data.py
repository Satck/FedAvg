# src/data/mnist_data.py

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_mnist_data(data_dir='./data/mnist'):
    """
    加载MNIST数据集
    
    Returns:
        train_dataset: 训练集
        test_dataset: 测试集
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset


def get_test_loader(test_dataset, batch_size=128):
    """创建测试集加载器"""
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


def create_iid_partition(dataset, num_clients=100, seed=42):
    """
    创建IID数据划分
    
    Args:
        dataset: PyTorch Dataset对象
        num_clients: 客户端数量
        seed: 随机种子
        
    Returns:
        client_indices: 字典 {client_id: [sample_indices]}
    """
    np.random.seed(seed)
    
    num_items = len(dataset) // num_clients
    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices)
    
    client_indices = {}
    for i in range(num_clients):
        start = i * num_items
        end = start + num_items
        client_indices[i] = all_indices[start:end]
    
    return client_indices


def create_client_loaders(dataset, client_indices, batch_size=10):
    """
    创建客户端数据加载器
    
    Args:
        dataset: 原始数据集
        client_indices: 客户端数据索引字典
        batch_size: batch大小
        
    Returns:
        client_loaders: 列表，每个元素是一个DataLoader
    """
    client_loaders = []
    
    for client_id in sorted(client_indices.keys()):
        indices = client_indices[client_id]
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # 避免多进程问题
        )
        client_loaders.append(loader)
    
    return client_loaders


def create_noniid_partition(dataset, num_clients=100, shards_per_client=2, seed=42):
    """
    创建Non-IID数据划分（论文方法）
    
    Args:
        dataset: PyTorch Dataset对象
        num_clients: 客户端数量
        shards_per_client: 每个客户端的碎片数
        seed: 随机种子
        
    Returns:
        client_indices: 字典 {client_id: [sample_indices]}
    """
    np.random.seed(seed)
    
    # 步骤1: 获取所有样本的标签并排序
    num_samples = len(dataset)
    labels = np.array([dataset[i][1] for i in range(num_samples)])
    sorted_indices = np.argsort(labels)
    
    # 步骤2: 分成碎片
    num_shards = num_clients * shards_per_client
    shard_size = num_samples // num_shards
    shards = []
    
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size
        shards.append(sorted_indices[start:end].tolist())
    
    # 步骤3: 随机打乱碎片顺序
    np.random.shuffle(shards)
    
    # 步骤4: 为每个客户端分配碎片
    client_indices = {}
    for i in range(num_clients):
        client_shards = shards[i * shards_per_client: (i + 1) * shards_per_client]
        client_indices[i] = []
        for shard in client_shards:
            client_indices[i].extend(shard)
    
    return client_indices


def verify_iid_partition(dataset, client_indices):
    """验证IID划分的正确性"""
    print("验证IID划分...")
    
    # 检查样本总数
    total_samples = sum(len(indices) for indices in client_indices.values())
    print(f"总样本数: {total_samples} (应为60000)")
    
    # 检查每个客户端样本数
    samples_per_client = [len(indices) for indices in client_indices.values()]
    print(f"每客户端样本数 - 最小: {min(samples_per_client)}, "
          f"最大: {max(samples_per_client)}, 平均: {np.mean(samples_per_client):.1f}")
    
    # 检查标签分布（前5个客户端）
    for client_id in range(5):
        indices = client_indices[client_id]
        labels = [dataset[i][1] for i in indices]
        label_counts = np.bincount(labels, minlength=10)
        print(f"客户端 {client_id} 标签分布: {label_counts}")


def verify_noniid_partition(dataset, client_indices):
    """验证Non-IID划分的正确性"""
    print("验证Non-IID划分...")
    
    # 检查样本总数
    total_samples = sum(len(indices) for indices in client_indices.values())
    print(f"总样本数: {total_samples} (应为60000)")
    
    # 统计每个客户端包含的不同数字类别数
    unique_labels_count = []
    for client_id in range(min(10, len(client_indices))):
        indices = client_indices[client_id]
        labels = [dataset[i][1] for i in indices]
        unique_labels = len(set(labels))
        unique_labels_count.append(unique_labels)
        
        label_counts = np.bincount(labels, minlength=10)
        print(f"客户端 {client_id} 标签分布: {label_counts} | 唯一标签数: {unique_labels}")
    
    print(f"平均每客户端唯一标签数: {np.mean(unique_labels_count):.2f} (应接近2)")


if __name__ == '__main__':
    # 测试代码
    train_dataset, test_dataset = load_mnist_data()
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 测试IID划分
    iid_indices = create_iid_partition(train_dataset)
    verify_iid_partition(train_dataset, iid_indices)
    
    # 测试Non-IID划分
    noniid_indices = create_noniid_partition(train_dataset)
    verify_noniid_partition(train_dataset, noniid_indices)
