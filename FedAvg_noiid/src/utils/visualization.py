# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_learning_curves(results_dict, save_path=None):
    """绘制学习曲线"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率曲线
    for name, data in results_dict.items():
        ax1.plot(data['history']['rounds'], 
                data['history']['test_acc'],
                label=name, linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Communication Rounds', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy vs Rounds', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线
    for name, data in results_dict.items():
        ax2.plot(data['history']['rounds'], 
                data['history']['test_loss'],
                label=name, linewidth=2, marker='o', markersize=4)
    
    ax2.set_xlabel('Communication Rounds', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Test Loss vs Rounds', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习曲线已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_selection_distribution(selection_stats_dict, save_path=None):
    """绘制客户端选择分布"""
    
    num_dists = len(selection_stats_dict)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (name, stats) in enumerate(selection_stats_dict.items()):
        if idx >= 6:
            break
        
        ax = axes[idx]
        counts = stats['selection_counts']
        
        ax.bar(range(len(counts)), counts, alpha=0.7)
        ax.set_xlabel('Client ID', fontsize=10)
        ax.set_ylabel('Selection Count', fontsize=10)
        ax.set_title(f'{name}\nMean: {stats["mean"]:.1f}, Std: {stats["std"]:.1f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for idx in range(num_dists, 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"选择分布图已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_summary_table(results_dict):
    """生成汇总表格"""
    
    print("\n" + "="*100)
    print("实验结果汇总")
    print("="*100)
    print(f"{'分布':<15} {'最终准确率':<12} {'选择均值':<12} {'选择标准差':<12} "
          f"{'最小选择':<12} {'最大选择':<12} {'最大/最小比':<12}")
    print("-"*100)
    
    for name, data in results_dict.items():
        final_acc = data['history']['test_acc'][-1]
        stats = data['selection_stats']
        max_min_ratio = stats['max'] / max(stats['min'], 1)
        
        print(f"{name:<15} {final_acc:<12.4f} {stats['mean']:<12.1f} "
              f"{stats['std']:<12.1f} {stats['min']:<12} {stats['max']:<12} "
              f"{max_min_ratio:<12.2f}")
    
    print("="*100)
