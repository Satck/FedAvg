# experiments/run_distribution_comparison.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from src.data.mnist_data import (
    load_mnist_data,
    create_iid_partition,
    create_client_loaders,
    get_test_loader
)
from src.models.cnn_model import MNIST_CNN
from src.client_selection.uniform_selector import UniformSelector
from src.client_selection.binomial_selector import BinomialSelector
from src.client_selection.poisson_selector import PoissonSelector
from src.client_selection.normal_selector import NormalSelector
from src.client_selection.exponential_selector import ExponentialSelector
from src.algorithms.fedavg import FederatedAveraging


def run_single_experiment(selector_name, selector_config, base_config, 
                          train_dataset, test_loader, client_indices):
    """运行单个实验"""
    
    print(f"\n{'='*80}")
    print(f"实验：{selector_name}")
    print(f"配置：{selector_config}")
    print(f"{'='*80}")
    
    # 设置随机种子
    seed = base_config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建客户端加载器
    client_loaders = create_client_loaders(
        train_dataset,
        client_indices,
        batch_size=base_config['federated']['batch_size']
    )
    
    # 创建模型
    model = MNIST_CNN()
    
    # 创建选择器
    selector_classes = {
        'uniform': UniformSelector,
        'binomial': BinomialSelector,
        'poisson': PoissonSelector,
        'normal': NormalSelector,
        'exponential': ExponentialSelector
    }
    
    selector = selector_classes[selector_name](
        num_clients=base_config['data']['num_clients'],
        config=selector_config
    )
    
    # 创建FedAvg
    fed_config = {
        'clients_per_round': base_config['federated']['clients_per_round'],
        'local_epochs': base_config['federated']['local_epochs'],
        'learning_rate': base_config['training']['learning_rate'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    fed_alg = FederatedAveraging(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        client_selector=selector,
        config=fed_config
    )
    
    # 训练
    history, selection_stats = fed_alg.train(
        num_rounds=base_config['training']['num_rounds'],
        eval_every=base_config['training']['eval_every']
    )
    
    return history, selection_stats


def plot_comparison(results_dict, save_path='results/figures/distribution_comparison.png'):
    """绘制对比图"""
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 子图1: 准确率对比
    ax1 = axes[0, 0]
    for name, data in results_dict.items():
        ax1.plot(data['history']['rounds'], data['history']['test_acc'], 
                label=name, linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Communication Rounds', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 损失对比
    ax2 = axes[0, 1]
    for name, data in results_dict.items():
        ax2.plot(data['history']['rounds'], data['history']['test_loss'], 
                label=name, linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Communication Rounds', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 客户端选择频率对比（bar chart）
    ax3 = axes[0, 2]
    dist_names = list(results_dict.keys())
    selection_stds = [results_dict[name]['selection_stats']['std'] for name in dist_names]
    
    ax3.bar(range(len(dist_names)), selection_stds, alpha=0.7)
    ax3.set_xticks(range(len(dist_names)))
    ax3.set_xticklabels(dist_names, rotation=45, ha='right')
    ax3.set_ylabel('Selection Std Dev', fontsize=12)
    ax3.set_title('Selection Fairness (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4-8: 每种分布的客户端选择频率分布
    for idx, (name, data) in enumerate(results_dict.items()):
        if idx >= 3:
            break
        
        ax = axes[1, idx]
        
        stats = data['selection_stats']
        counts = stats['selection_counts']
        
        # 绘制直方图
        ax.bar(range(len(counts)), counts, alpha=0.7, width=1.0)
        ax.set_xlabel('Client ID', fontsize=10)
        ax.set_ylabel('Selection Count', fontsize=10)
        ax.set_title(f'{name}\nMean: {stats["mean"]:.1f}, Std: {stats["std"]:.1f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存至: {save_path}")
    plt.close()


def main():
    """主函数"""
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'base_config.yaml')
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    print("="*80)
    print("实验：客户端选择概率分布对比")
    print("="*80)
    
    # 加载数据
    print("\n1. 加载数据...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', base_config['data']['data_dir'])
    train_dataset, test_dataset = load_mnist_data(data_dir)
    test_loader = get_test_loader(test_dataset)
    
    # 创建数据划分（IID）
    print("2. 创建IID数据划分...")
    client_indices = create_iid_partition(
        train_dataset,
        num_clients=base_config['data']['num_clients'],
        seed=base_config['experiment']['seed']
    )
    
    # 实验列表
    experiments = [
        ('uniform', {}),
        ('binomial', {'alpha': 2, 'beta': 5, 'static': True}),
        ('poisson', {'lambda': 5, 'static': True}),
        ('normal', {'sigma': 1.0}),
        ('exponential', {'lambda': 1.0})
    ]
    
    # 运行所有实验
    results = {}
    for selector_name, selector_config in experiments:
        history, selection_stats = run_single_experiment(
            selector_name, 
            selector_config, 
            base_config,
            train_dataset,
            test_loader,
            client_indices
        )
        
        results[selector_name] = {
            'history': history,
            'selection_stats': selection_stats
        }
    
    # 绘制对比图
    print("\n3. 生成对比图...")
    save_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'distribution_comparison.png')
    plot_comparison(results, save_path)
    
    # 打印汇总表
    print("\n" + "="*80)
    print("实验汇总")
    print("="*80)
    print(f"{'分布':<15} {'最终准确率':<12} {'选择均值':<12} {'选择标准差':<12} {'最大/最小':<12}")
    print("-"*80)
    for name, data in results.items():
        final_acc = data['history']['test_acc'][-1]
        stats = data['selection_stats']
        max_min_ratio = stats['max'] / max(stats['min'], 1)
        print(f"{name:<15} {final_acc:<12.4f} {stats['mean']:<12.1f} "
              f"{stats['std']:<12.1f} {max_min_ratio:<12.2f}")
    print("="*80)
    
    # 保存结果
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'distribution_comparison_results.pt')
    torch.save(results, results_path)
    print(f"\n结果已保存至: {results_path}")


if __name__ == '__main__':
    main()
