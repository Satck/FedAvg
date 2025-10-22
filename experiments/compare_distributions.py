# experiments/compare_distributions.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import pandas as pd
from pathlib import Path
from datetime import datetime


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_results(results_dir='../results'):
    """加载所有分布的实验结果"""
    results_dir = Path(results_dir)
    distributions = ['uniform', 'binomial', 'poisson', 'normal', 'exponential']
    
    all_results = {}
    missing_results = []
    
    for dist in distributions:
        result_file = results_dir / f'{dist}_results.pt'
        if result_file.exists():
            try:
                results = torch.load(result_file, map_location='cpu')
                all_results[dist] = results
                print(f"✅ 成功加载 {dist} 分布结果")
            except Exception as e:
                print(f"❌ 加载 {dist} 分布结果失败: {e}")
                missing_results.append(dist)
        else:
            print(f"⚠️  未找到 {dist} 分布结果文件: {result_file}")
            missing_results.append(dist)
    
    if missing_results:
        print(f"\n📋 缺失的结果: {missing_results}")
        print("请先运行相应的实验生成这些结果")
    
    return all_results, missing_results


def create_performance_comparison(all_results, save_dir='../results/figures'):
    """创建性能对比图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 颜色方案
    colors = {
        'uniform': '#2E86AB',      # 蓝色
        'binomial': '#A23B72',     # 紫红色
        'poisson': '#F18F01',      # 橙色
        'normal': '#C73E1D',       # 红色
        'exponential': '#5D737E'   # 灰蓝色
    }
    
    # 创建大图，包含多个子图
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('联邦学习客户端选择分布对比实验结果', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. 准确率对比
    ax1 = axes[0, 0]
    for dist_name, results in all_results.items():
        history = results['history']
        rounds = history['rounds']
        test_acc = history['test_acc']
        ax1.plot(rounds, test_acc, label=f'{dist_name.capitalize()}', 
                color=colors[dist_name], linewidth=2, marker='o', markersize=3)
    
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('测试准确率')
    ax1.set_title('测试准确率 vs 训练轮数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 损失对比
    ax2 = axes[0, 1]
    for dist_name, results in all_results.items():
        history = results['history']
        rounds = history['rounds']
        test_loss = history['test_loss']
        ax2.plot(rounds, test_loss, label=f'{dist_name.capitalize()}', 
                color=colors[dist_name], linewidth=2, marker='s', markersize=3)
    
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('测试损失')
    ax2.set_title('测试损失 vs 训练轮数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 最终性能柱状图
    ax3 = axes[0, 2]
    final_accuracies = []
    final_losses = []
    dist_names = []
    
    for dist_name, results in all_results.items():
        history = results['history']
        final_accuracies.append(history['test_acc'][-1])
        final_losses.append(history['test_loss'][-1])
        dist_names.append(dist_name.capitalize())
    
    x_pos = np.arange(len(dist_names))
    bars = ax3.bar(x_pos, final_accuracies, color=[colors[d.lower()] for d in dist_names], alpha=0.8)
    ax3.set_xlabel('分布类型')
    ax3.set_ylabel('最终准确率')
    ax3.set_title('最终准确率对比')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(dist_names, rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. 客户端选择公平性
    ax4 = axes[1, 0]
    fairness_ratios = []
    for dist_name, results in all_results.items():
        stats = results['selection_stats']
        fairness_ratio = stats['max'] / max(stats['min'], 1)
        fairness_ratios.append(fairness_ratio)
    
    bars = ax4.bar(x_pos, fairness_ratios, color=[colors[d.lower()] for d in dist_names], alpha=0.8)
    ax4.set_xlabel('分布类型')
    ax4.set_ylabel('公平性比率 (Max/Min)')
    ax4.set_title('客户端选择公平性 (越接近1越公平)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dist_names, rotation=45)
    
    # 添加公平性基准线
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='完全公平')
    ax4.legend()
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 5. 选择标准差对比
    ax5 = axes[1, 1]
    selection_stds = []
    for dist_name, results in all_results.items():
        stats = results['selection_stats']
        selection_stds.append(stats['std'])
    
    bars = ax5.bar(x_pos, selection_stds, color=[colors[d.lower()] for d in dist_names], alpha=0.8)
    ax5.set_xlabel('分布类型')
    ax5.set_ylabel('选择标准差')
    ax5.set_title('客户端选择标准差 (越小越均匀)')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(dist_names, rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 6. 训练耗时对比
    ax6 = axes[1, 2]
    training_times = []
    for dist_name, results in all_results.items():
        training_times.append(results['training_duration'] / 60)  # 转换为分钟
    
    bars = ax6.bar(x_pos, training_times, color=[colors[d.lower()] for d in dist_names], alpha=0.8)
    ax6.set_xlabel('分布类型')
    ax6.set_ylabel('训练时间 (分钟)')
    ax6.set_title('训练时间对比')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(dist_names, rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}min', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'distribution_comparison_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"📊 对比图表已保存: {filepath}")
    
    return filepath


def create_client_selection_heatmap(all_results, save_dir='../results/figures'):
    """创建客户端选择热图"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_distributions = len(all_results)
    fig, axes = plt.subplots(1, num_distributions, figsize=(4*num_distributions, 6))
    
    if num_distributions == 1:
        axes = [axes]
    
    fig.suptitle('客户端选择频次热图', fontsize=16, fontweight='bold')
    
    for idx, (dist_name, results) in enumerate(all_results.items()):
        stats = results['selection_stats']
        selection_counts = stats['selection_counts']
        
        # 将选择次数重塑为矩阵形式（假设客户端按网格排列）
        num_clients = len(selection_counts)
        grid_size = int(np.sqrt(num_clients))
        if grid_size * grid_size < num_clients:
            grid_size += 1
        
        # 填充到完整的网格
        padded_counts = np.zeros(grid_size * grid_size)
        padded_counts[:num_clients] = selection_counts
        heatmap_data = padded_counts.reshape(grid_size, grid_size)
        
        # 创建热图
        im = axes[idx].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        axes[idx].set_title(f'{dist_name.capitalize()}\n(Total: {stats["total_rounds"]} rounds)')
        axes[idx].set_xlabel('客户端列')
        axes[idx].set_ylabel('客户端行')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.set_label('选择次数')
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'client_selection_heatmap_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"🔥 选择热图已保存: {filepath}")
    
    return filepath


def create_detailed_report(all_results, save_dir='../results'):
    """生成详细报告"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(save_dir, f'comparison_report_{timestamp}.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("联邦学习客户端选择分布对比实验详细报告\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"对比分布数量: {len(all_results)}\n\n")
        
        # 整体结果摘要
        f.write("📊 整体结果摘要\n")
        f.write("-"*50 + "\n")
        
        summary_data = []
        for dist_name, results in all_results.items():
            history = results['history']
            stats = results['selection_stats']
            
            summary_data.append({
                'Distribution': dist_name.capitalize(),
                'Final_Accuracy': history['test_acc'][-1],
                'Final_Loss': history['test_loss'][-1],
                'Training_Time_Min': results['training_duration'] / 60,
                'Fairness_Ratio': stats['max'] / max(stats['min'], 1),
                'Selection_Std': stats['std']
            })
        
        # 创建表格
        df = pd.DataFrame(summary_data)
        f.write(df.to_string(index=False, float_format='%.4f'))
        f.write("\n\n")
        
        # 详细分析
        f.write("🔍 详细分析\n")
        f.write("-"*50 + "\n")
        
        for dist_name, results in all_results.items():
            f.write(f"\n📈 {dist_name.upper()} 分布\n")
            f.write("-"*30 + "\n")
            
            history = results['history']
            stats = results['selection_stats']
            config = results['config']
            
            # 性能指标
            f.write(f"性能指标:\n")
            f.write(f"  - 最终准确率: {history['test_acc'][-1]:.4f}\n")
            f.write(f"  - 最终损失: {history['test_loss'][-1]:.4f}\n")
            f.write(f"  - 最佳准确率: {max(history['test_acc']):.4f}\n")
            f.write(f"  - 最低损失: {min(history['test_loss']):.4f}\n")
            
            if len(history['test_acc']) > 1:
                improvement = history['test_acc'][-1] - history['test_acc'][0]
                f.write(f"  - 准确率提升: {improvement:+.4f}\n")
            
            # 公平性指标
            f.write(f"公平性指标:\n")
            f.write(f"  - 平均选择次数: {stats['mean']:.1f}\n")
            f.write(f"  - 选择标准差: {stats['std']:.2f}\n")
            f.write(f"  - 最大选择次数: {stats['max']}\n")
            f.write(f"  - 最小选择次数: {stats['min']}\n")
            f.write(f"  - 公平性比率: {stats['max'] / max(stats['min'], 1):.2f}\n")
            
            # 配置信息
            f.write(f"分布配置:\n")
            for key, value in config.items():
                f.write(f"  - {key}: {value}\n")
            
            # 时间统计
            f.write(f"时间统计:\n")
            f.write(f"  - 训练时间: {results['training_duration']:.1f} 秒 ({results['training_duration']/60:.1f} 分钟)\n")
            f.write(f"  - 总耗时: {results['total_duration']:.1f} 秒 ({results['total_duration']/60:.1f} 分钟)\n")
        
        # 排名分析
        f.write(f"\n🏆 排名分析\n")
        f.write("-"*50 + "\n")
        
        # 按最终准确率排名
        acc_ranking = sorted(summary_data, key=lambda x: x['Final_Accuracy'], reverse=True)
        f.write("最终准确率排名:\n")
        for i, item in enumerate(acc_ranking, 1):
            f.write(f"  {i}. {item['Distribution']}: {item['Final_Accuracy']:.4f}\n")
        
        # 按公平性排名（越接近1越好）
        fairness_ranking = sorted(summary_data, key=lambda x: abs(x['Fairness_Ratio'] - 1))
        f.write("\n公平性排名 (越接近1越公平):\n")
        for i, item in enumerate(fairness_ranking, 1):
            f.write(f"  {i}. {item['Distribution']}: {item['Fairness_Ratio']:.2f}\n")
        
        # 按训练效率排名
        time_ranking = sorted(summary_data, key=lambda x: x['Training_Time_Min'])
        f.write("\n训练效率排名 (时间越短越好):\n")
        for i, item in enumerate(time_ranking, 1):
            f.write(f"  {i}. {item['Distribution']}: {item['Training_Time_Min']:.1f} 分钟\n")
    
    print(f"📋 详细报告已保存: {report_file}")
    
    return report_file


def main():
    """主函数"""
    print("🔍 开始对比分析...")
    
    # 加载结果
    all_results, missing_results = load_results()
    
    if not all_results:
        print("❌ 没有找到任何实验结果文件")
        print("请先运行实验生成结果文件")
        return
    
    if len(all_results) < 2:
        print("⚠️  只有1个实验结果，无法进行对比")
        print("请运行更多实验以便对比")
        return
    
    print(f"\n✅ 成功加载 {len(all_results)} 个实验结果")
    print(f"包含分布: {list(all_results.keys())}")
    
    # 创建图表目录
    figures_dir = '../results/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    try:
        # 生成对比图表
        print("\n📊 生成性能对比图表...")
        performance_chart = create_performance_comparison(all_results)
        
        # 生成选择热图
        print("\n🔥 生成客户端选择热图...")
        heatmap_chart = create_client_selection_heatmap(all_results)
        
        # 生成详细报告
        print("\n📋 生成详细报告...")
        report_file = create_detailed_report(all_results)
        
        print("\n" + "="*80)
        print("🎉 对比分析完成！")
        print("="*80)
        print(f"📊 性能对比图: {performance_chart}")
        print(f"🔥 选择热图: {heatmap_chart}")
        print(f"📋 详细报告: {report_file}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
