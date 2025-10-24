#!/usr/bin/env python3
"""
Non-IID联邦学习实验可视化工具
改进版：专门处理Non-IID实验结果，展示更明显的分布差异
"""

import re
import matplotlib
matplotlib.use('Agg')  # 设置为非GUI后端
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
    """加载所有可用分布的Non-IID数据"""
    
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return {}
    
    distributions = {}
    distribution_names = ['uniform', 'binomial', 'poisson', 'normal', 'exponential']
    
    print("🔍 扫描Non-IID实验结果...")
    
    for dist_name in distribution_names:
        # 查找该分布的Non-IID日志文件
        log_files = list(logs_path.glob(f"{dist_name}_noiid_*.log"))
        
        if log_files:
            # 选择最新的日志文件
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"   📊 发现 {dist_name}_noiid: {latest_log.name}")
            
            # 解析数据
            metrics = parse_training_log(latest_log)
            if metrics:
                distributions[dist_name] = metrics
            else:
                print(f"   ⚠️  解析失败 {dist_name}_noiid")
        else:
            print(f"   ❌ 未找到 {dist_name}_noiid 数据")
    
    return distributions


def print_data_validation(distributions):
    """验证并打印各分布数据的差异"""
    
    print(f"\n🔍 Non-IID数据验证和差异分析:")
    print("-" * 60)
    
    # 分析准确率差异
    print("📊 准确率差异分析:")
    for dist_name, metrics in distributions.items():
        acc = metrics['accuracy']
        print(f"   {dist_name:>12}: 最终={acc[-1]:.4f}, 最佳={max(acc):.4f}, 标准差={np.std(acc):.6f}")
    
    # 比较第1轮、中间轮次和最后轮次的差异
    if len(distributions) > 1:
        rounds_to_check = [0, len(list(distributions.values())[0]['accuracy'])//2, -1]  # 第1轮、中间、最后
        round_labels = ["第1轮", "中间轮", "最后轮"]
        
        for i, round_idx in enumerate(rounds_to_check):
            print(f"\n📈 {round_labels[i]}准确率差异:")
            values = []
            for dist_name, metrics in distributions.items():
                acc_val = metrics['accuracy'][round_idx]
                values.append(acc_val)
                print(f"   {dist_name:>12}: {acc_val:.4f}")
            
            if len(values) > 1:
                diff_range = max(values) - min(values)
                print(f"   {'差异范围':>12}: {diff_range:.6f} ({diff_range*100:.4f}%)")


def plot_comparison(distributions, save_dir='results/figures'):
    """生成Non-IID实验对比图表"""
    
    if not distributions:
        print("❌ No distribution data to plot")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 首先打印数据验证信息
    print_data_validation(distributions)
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 定义颜色方案 - 更鲜明的颜色对比
    colors = {
        'uniform': '#1f77b4',      # 蓝色
        'binomial': '#ff7f0e',     # 橙色  
        'poisson': '#2ca02c',      # 绿色
        'normal': '#d62728',       # 红色
        'exponential': '#9467bd'   # 紫色
    }
    
    print(f"\n📊 绘制 {len(distributions)} 个Non-IID分布: {list(distributions.keys())}")
    
    # 收集所有准确率数据以优化y轴范围
    all_accuracies = []
    for metrics in distributions.values():
        all_accuracies.extend(metrics['accuracy'])
    
    acc_min, acc_max = min(all_accuracies), max(all_accuracies)
    acc_range = acc_max - acc_min
    
    print(f"📈 准确率范围: {acc_min:.4f} - {acc_max:.4f} (范围: {acc_range:.4f})")
    
    for dist_name, metrics in distributions.items():
        color = colors.get(dist_name, '#7f7f7f')  # 默认灰色
        
        # 1. Accuracy over rounds (左上) - 高亮显示差异
        axes[0, 0].plot(metrics['round'], metrics['accuracy'], 
                       label=dist_name, linewidth=3, color=color, marker='o', markersize=2, alpha=0.9)
        
        # 2. Loss over rounds (右上)
        axes[0, 1].plot(metrics['round'], metrics['loss'], 
                       label=dist_name, linewidth=2.5, color=color, alpha=0.9)
        
        # 3. Accuracy over time (左下) - 也高亮显示差异
        axes[1, 0].plot(metrics['wall_clock_time'], metrics['accuracy'], 
                       label=dist_name, linewidth=2.5, color=color, alpha=0.9)
        
        # 4. Training time per round (右下)
        axes[1, 1].plot(metrics['round'], metrics['training_time'], 
                       label=dist_name, linewidth=2, color=color, alpha=0.8)
    
    # 设置标题和标签 - 强调Non-IID
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Accuracy vs Round (Non-IID数据)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 为准确率图设置更合适的y轴范围
    if acc_range > 0:
        # 如果差异很小，放大显示
        if acc_range < 0.01:  # 小于1%差异
            margin = max(0.005, acc_range * 0.2)  # 至少0.5%的边距
            axes[0, 0].set_ylim(acc_min - margin, acc_max + margin)
            axes[0, 0].set_title('Accuracy vs Round (Non-IID, 放大显示)')
        # 如果差异较大，正常显示
        elif acc_range > 0.05:  # 大于5%差异
            axes[0, 0].set_ylim(0, 1)
    
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].set_title('Loss vs Round (Non-IID数据)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Wall-clock Time (s)')
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_title('Accuracy vs Time (Non-IID数据)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    # 为时间-准确率图也设置相同的y轴范围
    if acc_range > 0 and acc_range < 0.01:
        margin = max(0.005, acc_range * 0.2)
        axes[1, 0].set_ylim(acc_min - margin, acc_max + margin)
        axes[1, 0].set_title('Accuracy vs Time (Non-IID, 放大显示)')
    
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Training Time (s)')
    axes[1, 1].set_title('Training Time per Round')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 设置总标题
    plt.suptitle('Federated Learning - Non-IID Distribution Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/federated_noiid_comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Non-IID对比图已保存: {save_path}")
    
    # 显示图片 (在服务器环境中注释掉)
    # plt.show()
    plt.close()
    
    return save_path


def print_summary(distributions):
    """打印Non-IID实验摘要信息"""
    
    print(f"\n{'='*80}")
    print("📊 NON-IID联邦学习实验摘要")
    print(f"{'='*80}")
    
    if not distributions:
        print("❌ 未找到完成的Non-IID实验")
        return
    
    print(f"🎯 完成的Non-IID分布: {len(distributions)}")
    print(f"📋 可用分布: {', '.join(distributions.keys())}")
    
    # 创建对比表格
    print(f"\n📈 Non-IID性能对比:")
    print("-" * 80)
    print(f"{'分布':<15} {'轮数':<8} {'最终准确率':<12} {'最佳准确率':<12} {'时间(分)':<10} {'收敛情况':<10}")
    print("-" * 80)
    
    best_acc_dist = ""
    best_acc_value = 0
    fastest_dist = ""
    fastest_time = float('inf')
    
    for dist_name, metrics in distributions.items():
        rounds = len(metrics['round'])
        final_acc = metrics['accuracy'][-1]
        best_acc = max(metrics['accuracy'])
        total_time = metrics['wall_clock_time'][-1] / 60  # 转换为分钟
        
        # 判断收敛情况
        if len(metrics['accuracy']) >= 20:
            last_20_percent = int(len(metrics['accuracy']) * 0.2)
            recent_std = np.std(metrics['accuracy'][-last_20_percent:])
            
            if recent_std < 0.001:
                convergence = "✅已收敛"
            elif recent_std < 0.005:
                convergence = "🟡接近收敛"
            else:
                convergence = "🔄仍在学习"
        else:
            convergence = "❓数据不足"
        
        print(f"{dist_name:<15} {rounds:<8} {final_acc:<12.4f} {best_acc:<12.4f} {total_time:<10.1f} {convergence:<10}")
        
        # 跟踪最佳性能
        if best_acc > best_acc_value:
            best_acc_value = best_acc
            best_acc_dist = dist_name
        
        if total_time < fastest_time:
            fastest_time = total_time
            fastest_dist = dist_name
    
    print("-" * 80)
    print(f"🏆 最佳准确率: {best_acc_dist} ({best_acc_value:.4f})")
    print(f"⚡ 最快训练: {fastest_dist} ({fastest_time:.1f} 分钟)")
    
    # Non-IID效果分析
    if len(distributions) > 1:
        all_final_accs = [metrics['accuracy'][-1] for metrics in distributions.values()]
        acc_range = max(all_final_accs) - min(all_final_accs)
        
        print(f"\n🔍 Non-IID效果分析:")
        print(f"   最终准确率差异范围: {acc_range:.6f} ({acc_range*100:.4f}%)")
        
        if acc_range > 0.02:  # 2%以上差异
            print("   🟢 Non-IID设置产生了明显的分布差异效果!")
        elif acc_range > 0.005:  # 0.5-2%差异
            print("   🟡 Non-IID设置产生了中等程度的差异")
        else:
            print("   🔴 Non-IID设置的差异效果仍然较小")
            print("   💡 建议: 进一步降低alpha参数或调整分布参数")
    
    print(f"{'='*80}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Visualize Non-IID federated learning experiments')
    parser.add_argument('--logs-dir', type=str, default='results/logs',
                       help='Directory containing log files')
    parser.add_argument('--save-dir', type=str, default='results/figures',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("🎨 Non-IID联邦学习可视化工具")
    print("=" * 60)
    print("📁 自动检测完成的Non-IID实验...")
    
    # 加载所有可用的Non-IID分布数据
    distributions = load_all_distributions(args.logs_dir)
    
    if not distributions:
        print("\n❌ 未找到完成的Non-IID实验!")
        print("💡 运行Non-IID实验:")
        print("   python run_uniform_noiid.py")
        print("   python run_binomial_noiid.py") 
        print("   python run_poisson_noiid.py")
        print("   等等...")
        return
    
    # 打印摘要
    print_summary(distributions)
    
    # 生成对比图表
    print(f"\n🎨 生成Non-IID对比图表...")
    plot_path = plot_comparison(distributions, args.save_dir)
    
    if plot_path:
        print(f"\n🎉 Non-IID可视化完成!")
        print(f"📊 对比图表: {plot_path}")
        print(f"📈 展示了 {len(distributions)} 个Non-IID分布")
        
        # 提示下一步
        remaining_dists = set(['uniform', 'binomial', 'poisson', 'normal', 'exponential']) - set(distributions.keys())
        if remaining_dists:
            print(f"\n💡 运行更多Non-IID分布:")
            for dist in sorted(remaining_dists):
                print(f"   python run_{dist}_noiid.py")
            print("   python visualize_noiid.py  # 更新图表")


if __name__ == '__main__':
    main()
