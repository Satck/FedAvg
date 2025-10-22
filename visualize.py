#!/usr/bin/env python3
"""
每次调用都会显示所有已完成分布的训练结果
使用方法: python visualize.py
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import os
import argparse


def parse_training_log(log_path):
    """解析训练日志文件"""
    
    rounds = []
    accuracies = []
    losses = []
    timestamps = []
    
    # 正则表达式模式
    round_pattern = r'Round\s+(\d+)\s+\|\s+Acc:\s+([\d.]+)\s+\|\s+Loss:\s+([\d.]+)'
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})'
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                round_match = re.search(round_pattern, line)
                if round_match:
                    timestamp_match = re.search(timestamp_pattern, line)
                    if timestamp_match:
                        timestamp_str = timestamp_match.group(1)
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        round_num = int(round_match.group(1))
                        accuracy = float(round_match.group(2))
                        loss = float(round_match.group(3))
                        
                        rounds.append(round_num)
                        accuracies.append(accuracy)
                        losses.append(loss)
                        timestamps.append(timestamp)
    
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None
    
    if not rounds:
        return None
    
    # 计算时间数据
    wall_clock_times = []
    training_times = []
    
    if timestamps:
        start_time = timestamps[0]
        
        # 累积时间
        for ts in timestamps:
            elapsed_seconds = (ts - start_time).total_seconds()
            wall_clock_times.append(elapsed_seconds)
        
        # 每轮训练时间
        for i in range(len(timestamps)):
            if i == 0:
                if len(timestamps) > 1:
                    avg_time = (timestamps[-1] - timestamps[0]).total_seconds() / len(timestamps)
                    training_times.append(avg_time)
                else:
                    training_times.append(0)
            else:
                round_time = (timestamps[i] - timestamps[i-1]).total_seconds()
                training_times.append(round_time)
    
    return {
        'round': rounds,
        'accuracy': accuracies,
        'loss': losses,
        'wall_clock_time': wall_clock_times,
        'training_time': training_times
    }


def load_all_distributions(logs_dir='results/logs'):
    """加载所有可用分布的数据"""
    
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return {}
    
    distributions = {}
    distribution_names = ['uniform', 'binomial', 'poisson', 'normal', 'exponential']
    
    print("🔍 Scanning for completed experiments...")
    
    for dist_name in distribution_names:
        # 查找该分布的最新日志文件
        log_files = list(logs_path.glob(f"{dist_name}_*.log"))
        
        if log_files:
            # 选择最新的日志文件
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"   📊 Found {dist_name}: {latest_log.name}")
            
            # 解析数据
            metrics = parse_training_log(latest_log)
            if metrics:
                distributions[dist_name] = metrics
            else:
                print(f"   ⚠️  Failed to parse {dist_name} data")
        else:
            print(f"   ❌ No data found for {dist_name}")
    
    return distributions


def plot_comparison(distributions, save_dir='results/figures'):

    
    if not distributions:
        print("❌ No distribution data to plot")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建2x2子图布局 - 与TIFL完全一致
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 定义颜色方案 - 每个分布一个颜色
    colors = {
        'uniform': '#1f77b4',      # 蓝色
        'binomial': '#ff7f0e',     # 橙色  
        'poisson': '#2ca02c',      # 绿色
        'normal': '#d62728',       # 红色
        'exponential': '#9467bd'   # 紫色
    }
    
    print(f"\n📊 Plotting {len(distributions)} distributions: {list(distributions.keys())}")
    
    for dist_name, metrics in distributions.items():
        color = colors.get(dist_name, '#7f7f7f')  # 默认灰色
        
        # 1. Accuracy over rounds (左上)
        axes[0, 0].plot(metrics['round'], metrics['accuracy'], 
                       label=dist_name, linewidth=2, color=color)
        
        # 2. Loss over rounds (右上)
        axes[0, 1].plot(metrics['round'], metrics['loss'], 
                       label=dist_name, linewidth=2, color=color)
        
        # 3. Accuracy over time (左下)
        axes[1, 0].plot(metrics['wall_clock_time'], metrics['accuracy'], 
                       label=dist_name, linewidth=2, color=color)
        
        # 4. Training time per round (右下)
        axes[1, 1].plot(metrics['round'], metrics['training_time'], 
                       label=dist_name, linewidth=2, color=color)
    
    # 设置标题和标签 - 完全按照TIFL格式
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Accuracy vs Round')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].set_title('Loss vs Round')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Wall-clock Time (s)')
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_title('Accuracy vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Training Time (s)')
    axes[1, 1].set_title('Training Time per Round')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 设置总标题
    plt.suptitle('Federated Learning - Distribution Comparison', fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/federated_comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to {save_path}")
    
    # 显示图片
    plt.show()
    plt.close()
    
    return save_path


def print_summary(distributions):
    """打印所有分布的摘要信息"""
    
    print(f"\n{'='*70}")
    print("📊 FEDERATED LEARNING EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    
    if not distributions:
        print("❌ No completed experiments found")
        return
    
    print(f"🎯 Completed Distributions: {len(distributions)}")
    print(f"📋 Available: {', '.join(distributions.keys())}")
    
    # 创建对比表格
    print(f"\n📈 Performance Comparison:")
    print("-" * 70)
    print(f"{'Distribution':<12} {'Rounds':<8} {'Final Acc':<12} {'Best Acc':<12} {'Time(min)':<10}")
    print("-" * 70)
    
    best_acc_dist = ""
    best_acc_value = 0
    fastest_dist = ""
    fastest_time = float('inf')
    
    for dist_name, metrics in distributions.items():
        rounds = len(metrics['round'])
        final_acc = metrics['accuracy'][-1]
        best_acc = max(metrics['accuracy'])
        total_time = metrics['wall_clock_time'][-1] / 60  # 转换为分钟
        
        print(f"{dist_name:<12} {rounds:<8} {final_acc:<12.4f} {best_acc:<12.4f} {total_time:<10.1f}")
        
        # 跟踪最佳性能
        if best_acc > best_acc_value:
            best_acc_value = best_acc
            best_acc_dist = dist_name
        
        if total_time < fastest_time:
            fastest_time = total_time
            fastest_dist = dist_name
    
    print("-" * 70)
    print(f"🏆 Best Accuracy: {best_acc_dist} ({best_acc_value:.4f})")
    print(f"⚡ Fastest Training: {fastest_dist} ({fastest_time:.1f} min)")
    
    # 收敛分析
    print(f"\n🎯 Convergence Analysis:")
    for dist_name, metrics in distributions.items():
        accuracies = metrics['accuracy']
        if len(accuracies) >= 20:
            # 计算后20%的标准差来判断收敛
            last_20_percent = int(len(accuracies) * 0.2)
            recent_std = np.std(accuracies[-last_20_percent:])
            
            if recent_std < 0.001:
                status = "✅ Converged"
            elif recent_std < 0.005:
                status = "🟡 Nearly Converged"
            else:
                status = "🔄 Still Learning"
            
            print(f"   {dist_name:<12}: {status} (std: {recent_std:.6f})")
    
    print(f"{'='*70}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Visualize federated learning experiments')
    parser.add_argument('--logs-dir', type=str, default='results/logs',
                       help='Directory containing log files')
    parser.add_argument('--save-dir', type=str, default='results/figures',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("🎨 Federated Learning Visualization Tool")
    print("=" * 50)
    print("📁 Automatically detecting completed experiments...")
    
    # 加载所有可用的分布数据
    distributions = load_all_distributions(args.logs_dir)
    
    if not distributions:
        print("\n❌ No completed experiments found!")
        print("💡 Run experiments first:")
        print("   python run_uniform.py")
        print("   python run_poisson.py") 
        print("   python run_binomial.py")
        print("   etc.")
        return
    
    # 打印摘要
    print_summary(distributions)
    
    # 生成对比图表
    print(f"\n🎨 Generating comparison plot...")
    plot_path = plot_comparison(distributions, args.save_dir)
    
    if plot_path:
        print(f"\n🎉 Visualization complete!")
        print(f"📊 Comparison plot: {plot_path}")
        print(f"📈 Showing {len(distributions)} distributions")
        
        # 提示下一步
        remaining_dists = set(['uniform', 'binomial', 'poisson', 'normal', 'exponential']) - set(distributions.keys())
        if remaining_dists:
            print(f"\n💡 To add more distributions, run:")
            for dist in sorted(remaining_dists):
                print(f"   python run_{dist}.py")
            print("   python visualize.py  # to update the plot")


if __name__ == '__main__':
    main()
