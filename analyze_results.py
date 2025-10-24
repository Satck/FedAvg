#!/usr/bin/env python3
"""
分析联邦学习实验结果 - 纯文本版本
不依赖matplotlib，专门用于分析曲线重合问题
"""

import re
from pathlib import Path
from datetime import datetime


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
    
    return {
        'round': rounds,
        'accuracy': accuracies,
        'loss': losses,
        'timestamps': timestamps
    }


def load_all_distributions(logs_dir='results/logs'):
    """加载所有可用分布的数据"""
    
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return {}
    
    distributions = {}
    distribution_names = ['uniform', 'binomial', 'poisson', 'normal', 'exponential']
    
    print("🔍 扫描已完成的实验...")
    
    for dist_name in distribution_names:
        # 查找该分布的最新日志文件
        log_files = list(logs_path.glob(f"{dist_name}_*.log"))
        
        if log_files:
            # 选择最新的日志文件
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"   📊 发现 {dist_name}: {latest_log.name}")
            
            # 解析数据
            metrics = parse_training_log(latest_log)
            if metrics:
                distributions[dist_name] = metrics
            else:
                print(f"   ⚠️  解析失败 {dist_name}")
        else:
            print(f"   ❌ 未找到数据 {dist_name}")
    
    return distributions


def analyze_accuracy_differences(distributions):
    """深入分析准确率差异"""
    
    print(f"\n{'='*80}")
    print("🔍 详细的准确率差异分析")
    print(f"{'='*80}")
    
    if len(distributions) < 2:
        print("❌ 需要至少2个分布进行比较")
        return
    
    # 1. 基本统计信息
    print("\n📊 基本统计信息:")
    print(f"{'分布':<12} {'轮数':<8} {'最终准确率':<12} {'最佳准确率':<12} {'标准差':<12} {'变化范围':<12}")
    print("-" * 80)
    
    final_accs = {}
    best_accs = {}
    
    for dist_name, metrics in distributions.items():
        acc = metrics['accuracy']
        final_acc = acc[-1]
        best_acc = max(acc)
        std_dev = sum((x - sum(acc)/len(acc))**2 for x in acc) / len(acc)  # 手动计算标准差
        std_dev = std_dev ** 0.5
        acc_range = max(acc) - min(acc)
        
        final_accs[dist_name] = final_acc
        best_accs[dist_name] = best_acc
        
        print(f"{dist_name:<12} {len(acc):<8} {final_acc:<12.6f} {best_acc:<12.6f} {std_dev:<12.6f} {acc_range:<12.6f}")
    
    # 2. 轮次差异分析
    print(f"\n🎯 关键轮次的准确率对比:")
    
    # 获取所有分布共同的轮数
    min_rounds = min(len(metrics['accuracy']) for metrics in distributions.values())
    
    key_rounds = [1, min_rounds//4, min_rounds//2, min_rounds*3//4, min_rounds]
    
    for round_idx in key_rounds:
        if round_idx <= 0 or round_idx > min_rounds:
            continue
            
        print(f"\n第 {round_idx} 轮:")
        round_accs = []
        for dist_name, metrics in distributions.items():
            acc_val = metrics['accuracy'][round_idx-1]  # 0-based index
            round_accs.append(acc_val)
            print(f"   {dist_name:>10}: {acc_val:.6f}")
        
        if len(round_accs) > 1:
            acc_min, acc_max = min(round_accs), max(round_accs)
            diff_range = acc_max - acc_min
            print(f"   {'差异范围':>10}: {diff_range:.6f} ({diff_range*100:.4f}%)")
    
    # 3. 收敛性分析
    print(f"\n🎯 收敛性分析 (最后20%轮次的稳定性):")
    for dist_name, metrics in distributions.items():
        accuracies = metrics['accuracy']
        if len(accuracies) >= 20:
            # 计算后20%的标准差
            last_20_percent = int(len(accuracies) * 0.2)
            recent_accs = accuracies[-last_20_percent:]
            
            # 手动计算标准差
            mean_acc = sum(recent_accs) / len(recent_accs)
            variance = sum((x - mean_acc)**2 for x in recent_accs) / len(recent_accs)
            recent_std = variance ** 0.5
            
            if recent_std < 0.001:
                status = "✅ 已收敛"
            elif recent_std < 0.005:
                status = "🟡 接近收敛"
            else:
                status = "🔄 仍在学习"
            
            print(f"   {dist_name:<12}: {status} (std: {recent_std:.6f})")


def analyze_learning_patterns(distributions):
    """分析学习模式"""
    
    print(f"\n🧠 学习模式分析:")
    print("-" * 50)
    
    for dist_name, metrics in distributions.items():
        accuracies = metrics['accuracy']
        
        if len(accuracies) < 10:
            continue
        
        # 计算学习速度（前10轮的改进率）
        early_improvement = accuracies[9] - accuracies[0] if len(accuracies) > 9 else 0
        
        # 计算后期稳定性（最后10轮的变化）
        if len(accuracies) >= 10:
            late_change = max(accuracies[-10:]) - min(accuracies[-10:])
        else:
            late_change = 0
        
        print(f"   {dist_name}:")
        print(f"      初始准确率: {accuracies[0]:.6f}")
        print(f"      前10轮改进: {early_improvement:.6f}")
        print(f"      后期稳定性: {late_change:.6f} (变化范围)")


def find_potential_causes(distributions):
    """分析可能的原因"""
    
    print(f"\n🔍 问题诊断:")
    print("-" * 50)
    
    # 检查数据相似性
    all_final_accs = [metrics['accuracy'][-1] for metrics in distributions.values()]
    
    if len(all_final_accs) > 1:
        acc_min, acc_max = min(all_final_accs), max(all_final_accs)
        total_range = acc_max - acc_min
        
        print(f"📊 最终准确率分布:")
        print(f"   最低: {acc_min:.6f}")
        print(f"   最高: {acc_max:.6f}")
        print(f"   范围: {total_range:.6f} ({total_range*100:.4f}%)")
        
        if total_range < 0.001:  # 0.1%差异
            print("\n❌ 问题确认: 所有分布的结果几乎相同!")
            print("可能的原因:")
            print("1. 🎲 所有实验使用了相同的随机种子")
            print("2. 📊 数据是IID分布，不同客户端选择策略影响很小")
            print("3. 🔧 模型初始化相同")
            print("4. ⚙️ 超参数设置相同")
            
            print(f"\n💡 建议的解决方案:")
            print("1. 使用不同的随机种子")
            print("2. 创建non-IID数据分布")
            print("3. 调整客户端选择策略的参数")
            print("4. 增加实验的差异性设置")
        
        elif total_range < 0.01:  # 1%差异
            print(f"\n⚠️ 差异较小但存在 ({total_range*100:.4f}%)")
            print("建议: 放大图表y轴范围以显示微小差异")
        
        else:
            print(f"\n✅ 发现明显差异 ({total_range*100:.4f}%)")


def main():
    """主函数"""
    
    print("🔍 联邦学习结果分析工具 (纯文本版)")
    print("="*60)
    
    # 加载所有可用的分布数据
    distributions = load_all_distributions('results/logs')
    
    if not distributions:
        print("\n❌ 未找到已完成的实验!")
        print("💡 请先运行实验:")
        print("   python run_uniform.py")
        print("   python run_binomial.py")
        print("   等等...")
        return
    
    print(f"\n✅ 成功加载 {len(distributions)} 个分布的数据:")
    for dist_name in distributions.keys():
        print(f"   📊 {dist_name}")
    
    # 执行详细分析
    analyze_accuracy_differences(distributions)
    analyze_learning_patterns(distributions)
    find_potential_causes(distributions)
    
    print(f"\n{'='*60}")
    print("🎉 分析完成!")
    print("📋 基于以上分析，您可以:")
    print("   1. 修改实验设置以产生更明显的差异")
    print("   2. 调整可视化代码以放大微小差异")
    print("   3. 使用non-IID数据分布")


if __name__ == '__main__':
    main()
