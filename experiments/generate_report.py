# experiments/generate_report.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.visualization import *


def generate_full_report():
    """生成完整实验报告"""
    
    print("="*80)
    print("生成实验报告")
    print("="*80)
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 加载结果
    print("\n1. 加载实验结果...")
    
    try:
        # IID基准
        iid_path = os.path.join(results_dir, 'iid_baseline_results.pt')
        if os.path.exists(iid_path):
            iid_results = torch.load(iid_path)
            print(f"   - IID基准: 最终准确率 {iid_results['history']['test_acc'][-1]:.4f}")
        else:
            print(f"   - IID基准结果文件不存在: {iid_path}")
            iid_results = None
        
        # Non-IID基准
        noniid_path = os.path.join(results_dir, 'noniid_baseline_results.pt')
        if os.path.exists(noniid_path):
            noniid_results = torch.load(noniid_path)
            print(f"   - Non-IID基准: 最终准确率 {noniid_results['history']['test_acc'][-1]:.4f}")
        else:
            print(f"   - Non-IID基准结果文件不存在: {noniid_path}")
            noniid_results = None
        
        # 分布对比
        dist_path = os.path.join(results_dir, 'distribution_comparison_results.pt')
        if os.path.exists(dist_path):
            dist_results = torch.load(dist_path)
            print(f"   - 分布对比: {len(dist_results)} 个实验")
        else:
            print(f"   - 分布对比结果文件不存在: {dist_path}")
            dist_results = None
    except Exception as e:
        print(f"   - 加载结果文件时出错: {e}")
        return
    
    # 生成可视化
    print("\n2. 生成可视化图表...")
    
    try:
        # IID vs Non-IID对比
        if iid_results and noniid_results:
            iid_noniid_dict = {
                'IID': iid_results,
                'Non-IID': noniid_results
            }
            plot_learning_curves(iid_noniid_dict, os.path.join(figures_dir, 'iid_vs_noniid.png'))
        
        # 分布对比
        if dist_results:
            plot_learning_curves(dist_results, os.path.join(figures_dir, 'distribution_learning_curves.png'))
            
            # 选择分布
            selection_stats_dict = {
                name: data['selection_stats'] 
                for name, data in dist_results.items()
            }
            plot_selection_distribution(selection_stats_dict, os.path.join(figures_dir, 'selection_distributions.png'))
            
            # 生成汇总表
            print("\n3. 生成汇总表...")
            generate_summary_table(dist_results)
        
    except Exception as e:
        print(f"   - 生成可视化时出错: {e}")
    
    print("\n" + "="*80)
    print("报告生成完成！")
    print(f"图表保存在: {figures_dir}")
    print("="*80)


if __name__ == '__main__':
    generate_full_report()
