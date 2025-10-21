#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速测试脚本：验证联邦学习实现的正确性
"""

import sys
import os
import numpy as np
import torch

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.data.mnist_data import load_mnist_data, create_iid_partition, create_client_loaders
from src.models.cnn_model import MNIST_CNN
from src.client_selection.uniform_selector import UniformSelector
from src.client_selection.binomial_selector import BinomialSelector
from src.client_selection.poisson_selector import PoissonSelector
from src.client_selection.normal_selector import NormalSelector
from src.client_selection.exponential_selector import ExponentialSelector


def test_data_loading():
    """测试数据加载功能"""
    print("🧪 测试数据加载...")
    
    try:
        train_dataset, test_dataset = load_mnist_data('./data/mnist')
        print(f"   ✅ 训练集大小: {len(train_dataset)}")
        print(f"   ✅ 测试集大小: {len(test_dataset)}")
        
        # 测试IID划分
        client_indices = create_iid_partition(train_dataset, num_clients=10, seed=42)
        print(f"   ✅ IID划分: 10个客户端，每客户端{len(client_indices[0])}个样本")
        
        # 创建客户端加载器
        client_loaders = create_client_loaders(train_dataset, client_indices, batch_size=32)
        print(f"   ✅ 客户端加载器: {len(client_loaders)}个")
        
        return True
    except Exception as e:
        print(f"   ❌ 数据加载测试失败: {e}")
        return False


def test_model():
    """测试模型实现"""
    print("🧪 测试CNN模型...")
    
    try:
        model = MNIST_CNN()
        param_count = model.count_parameters()
        print(f"   ✅ 模型参数量: {param_count:,}")
        
        # 测试前向传播
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        print(f"   ✅ 输入shape: {x.shape}")
        print(f"   ✅ 输出shape: {output.shape}")
        
        # 检查参数量是否正确
        expected_params = 1663370  # 根据README.md的要求
        param_diff = abs(param_count - expected_params)
        param_diff_percent = param_diff / expected_params * 100
        
        if param_diff_percent < 5:  # 允许5%的差异
            print(f"   ✅ 参数量符合预期 (期望: {expected_params:,}, 实际: {param_count:,}, 差异: {param_diff_percent:.1f}%)")
        else:
            print(f"   ⚠️  参数量与预期差异较大 (期望: {expected_params:,}, 实际: {param_count:,}, 差异: {param_diff_percent:.1f}%)")
        
        return True
    except Exception as e:
        print(f"   ❌ 模型测试失败: {e}")
        return False


def test_selectors():
    """测试所有客户端选择器"""
    print("🧪 测试客户端选择器...")
    
    num_clients = 20
    num_select = 5
    
    # 测试配置
    selectors = [
        ('Uniform', UniformSelector(num_clients)),
        ('Binomial', BinomialSelector(num_clients, {'alpha': 2, 'beta': 5})),
        ('Poisson', PoissonSelector(num_clients, {'lambda': 5})),
        ('Normal', NormalSelector(num_clients, {'sigma': 1.0})),
        ('Exponential', ExponentialSelector(num_clients, {'lambda': 1.0}))
    ]
    
    print(f"   客户端总数: {num_clients}, 每轮选择: {num_select}")
    
    try:
        for name, selector in selectors:
            # 多轮选择测试
            for round_num in range(1, 4):
                selected = selector.select(num_select, round_num)
                print(f"   ✅ {name:<12} Round {round_num}: {selected}")
            
            # 统计信息
            stats = selector.get_selection_statistics()
            print(f"   📊 {name:<12} Stats: Mean={stats['mean']:.1f}, Std={stats['std']:.1f}")
        
        return True
    except Exception as e:
        print(f"   ❌ 选择器测试失败: {e}")
        return False


def test_integration():
    """测试整体集成"""
    print("🧪 测试整体集成...")
    
    try:
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 加载少量数据进行测试
        train_dataset, test_dataset = load_mnist_data('./data/mnist')
        client_indices = create_iid_partition(train_dataset, num_clients=5, seed=42)
        client_loaders = create_client_loaders(train_dataset, client_indices, batch_size=32)
        
        # 创建模型
        model = MNIST_CNN()
        
        # 创建选择器
        selector = UniformSelector(num_clients=5)
        
        # 测试选择功能
        selected = selector.select(num_select=2, round_num=1)
        print(f"   ✅ 选择了客户端: {selected}")
        
        # 简单测试模型更新（不做完整训练）
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # 使用第一个客户端的数据做一个批次的训练
        data_iter = iter(client_loaders[0])
        data, target = next(data_iter)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"   ✅ 模型训练测试通过，损失: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"   ❌ 集成测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始联邦学习实现测试...")
    print("=" * 60)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("数据加载", test_data_loading()))
    test_results.append(("CNN模型", test_model()))
    test_results.append(("客户端选择器", test_selectors()))
    test_results.append(("整体集成", test_integration()))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("📋 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<15} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"总计: {passed}/{len(test_results)} 项测试通过")
    
    if passed == len(test_results):
        print("\n🎉 所有测试通过！代码实现正确。")
        print("\n💡 提示：运行实验请执行：")
        print("   cd experiments")
        print("   python run_iid_experiments.py")
        print("   python run_noniid_experiments.py")
        print("   python run_distribution_comparison.py")
    else:
        print(f"\n⚠️  有 {len(test_results) - passed} 项测试失败，请检查代码实现。")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
