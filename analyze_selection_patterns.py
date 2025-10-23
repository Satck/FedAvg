# analyze_selection_patterns.py
# 分析每个分布的客户端选择模式

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 对于统计摘要，使用等宽字体但避免中文字符问题
import matplotlib.font_manager as fm
try:
    # 尝试找到支持中文的字体
    chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'Arial Unicode MS' in f.name]
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = chinese_fonts + ['Helvetica', 'DejaVu Sans']
except:
    pass

def analyze_distribution(distribution_name):
    """分析单个分布的选择模式"""
    
    result_file = f"results/{distribution_name}_results.pt"
    
    if not os.path.exists(result_file):
        print(f"❌ 未找到结果文件: {result_file}")
        return None
    
    # 加载结果
    results = torch.load(result_file, weights_only=False)
    selection_stats = results['selection_stats']
    selection_counts = selection_stats['selection_counts']
    
    print(f"\n{'='*60}")
    print(f"📊 {distribution_name.upper()} 分布选择分析")
    print(f"{'='*60}")
    print(f"总轮数: {selection_stats['total_rounds']}")
    print(f"平均选择次数: {selection_stats['mean']:.2f}")
    print(f"标准差: {selection_stats['std']:.2f}")
    print(f"最大选择次数: {selection_stats['max']}")
    print(f"最小选择次数: {selection_stats['min']}")
    print(f"公平性比率 (max/min): {selection_stats['max']/max(selection_stats['min'],1):.2f}")
    
    # 统计选择次数分布
    unique, counts = np.unique(selection_counts.astype(int), return_counts=True)
    print(f"\n选择次数分布:")
    for times, count in zip(unique[:10], counts[:10]):  # 显示前10个
        print(f"  被选{times}次: {count}个客户端")
    
    # 检查是否有客户端从未被选中
    never_selected = np.sum(selection_counts == 0)
    if never_selected > 0:
        print(f"\n⚠️  警告: 有 {never_selected} 个客户端从未被选中!")
        print(f"未选中的客户端ID: {np.where(selection_counts == 0)[0].tolist()}")
    
    # 检查选择最多和最少的客户端
    most_selected = np.argsort(selection_counts)[-5:][::-1]
    least_selected = np.argsort(selection_counts)[:5]
    
    print(f"\n选择最多的5个客户端:")
    for client_id in most_selected:
        print(f"  客户端 {client_id}: 被选 {int(selection_counts[client_id])} 次")
    
    print(f"\n选择最少的5个客户端:")
    for client_id in least_selected:
        print(f"  客户端 {client_id}: 被选 {int(selection_counts[client_id])} 次")
    
    return {
        'name': distribution_name,
        'counts': selection_counts,
        'stats': selection_stats
    }


def visualize_selection_patterns(all_data):
    """可视化所有分布的选择模式"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('客户端选择模式对比分析', fontsize=16, fontweight='bold')
    
    colors = {
        'uniform': '#1f77b4',
        'binomial': '#ff7f0e', 
        'poisson': '#2ca02c',
        'normal': '#d62728',
        'exponential': '#9467bd'
    }
    
    # 1. 直方图：选择次数分布
    ax1 = axes[0, 0]
    for data in all_data:
        ax1.hist(data['counts'], bins=30, alpha=0.5, 
                label=data['name'], color=colors[data['name']])
    ax1.set_xlabel('选择次数')
    ax1.set_ylabel('客户端数量')
    ax1.set_title('选择次数分布直方图')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱线图：选择次数统计
    ax2 = axes[0, 1]
    box_data = [data['counts'] for data in all_data]
    box_labels = [data['name'] for data in all_data]
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, name in zip(bp['boxes'], box_labels):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax2.set_ylabel('选择次数')
    ax2.set_title('选择次数箱线图')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. 柱状图：公平性指标
    ax3 = axes[0, 2]
    fairness_ratios = [data['stats']['max']/max(data['stats']['min'],1) 
                       for data in all_data]
    bars = ax3.bar(range(len(all_data)), fairness_ratios, 
                   color=[colors[d['name']] for d in all_data])
    ax3.set_xticks(range(len(all_data)))
    ax3.set_xticklabels([d['name'] for d in all_data], rotation=45)
    ax3.set_ylabel('公平性比率 (max/min)')
    ax3.set_title('选择公平性对比 (越小越公平)')
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='完全公平')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, ratio) in enumerate(zip(bars, fairness_ratios)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. 折线图：按客户端ID的选择次数
    ax4 = axes[1, 0]
    for data in all_data:
        ax4.plot(range(len(data['counts'])), data['counts'], 
                label=data['name'], color=colors[data['name']], alpha=0.7)
    ax4.set_xlabel('客户端ID')
    ax4.set_ylabel('选择次数')
    ax4.set_title('各客户端选择次数曲线')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 热力图：选择次数（仅显示前50个客户端）
    ax5 = axes[1, 1]
    heatmap_data = np.array([data['counts'][:50] for data in all_data])
    im = ax5.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    ax5.set_yticks(range(len(all_data)))
    ax5.set_yticklabels([d['name'] for d in all_data])
    ax5.set_xlabel('客户端ID (前50个)')
    ax5.set_title('选择次数热力图')
    plt.colorbar(im, ax=ax5, label='选择次数')
    
    # 6. 统计摘要
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = "统计摘要\n" + "="*40 + "\n\n"
    for data in all_data:
        stats = data['stats']
        summary_text += f"{data['name'].upper()}:\n"
        summary_text += f"  均值: {stats['mean']:.2f}\n"
        summary_text += f"  标准差: {stats['std']:.2f}\n"
        summary_text += f"  范围: [{stats['min']}, {stats['max']}]\n"
        summary_text += f"  公平性: {stats['max']/max(stats['min'],1):.2f}\n\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='sans-serif',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "client_selection_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 分析图表已保存至: {output_file}")
    
    plt.show()


def main():
    """主函数"""
    print("="*60)
    print("🔍 客户端选择模式分析工具")
    print("="*60)
    
    distributions = ['uniform', 'binomial', 'poisson', 'normal', 'exponential']
    all_data = []
    
    for dist in distributions:
        data = analyze_distribution(dist)
        if data:
            all_data.append(data)
    
    if len(all_data) >= 2:
        print(f"\n{'='*60}")
        print("📊 开始生成可视化对比图...")
        print(f"{'='*60}")
        visualize_selection_patterns(all_data)
    else:
        print("\n⚠️  至少需要2个分布的结果才能进行对比分析")
    
    print("\n✅ 分析完成!")


if __name__ == '__main__':
    main()

