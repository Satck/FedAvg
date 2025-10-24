#!/usr/bin/env python3
"""
运行所有Non-IID分布实验
按顺序执行所有5个分布的联邦学习实验
"""

import subprocess
import sys
import time
from datetime import datetime


def run_experiment(script_name, distribution_name):
    """运行单个实验脚本"""
    print(f"\n{'='*60}")
    print(f"🚀 开始运行 {distribution_name.upper()} 分布实验")
    print(f"📝 脚本: {script_name}")
    print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行实验脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              cwd='.')
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {distribution_name.upper()} 实验完成成功!")
            print(f"⏱️ 耗时: {duration/60:.1f} 分钟")
        else:
            print(f"\n❌ {distribution_name.upper()} 实验失败!")
            print(f"❌ 错误代码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n💥 运行 {distribution_name.upper()} 实验时出错: {e}")
        return False
    
    return True


def main():
    """主函数 - 按顺序运行所有实验"""
    
    print("🎯 Non-IID联邦学习批量实验工具")
    print("=" * 60)
    print("📊 将按顺序运行5个分布的Non-IID实验")
    print("🎲 每个分布使用不同的随机种子")
    print("📈 使用Non-IID数据分布 (alpha=0.3)")
    
    # 实验配置
    experiments = [
        ('run_uniform_noiid.py', 'uniform'),
        ('run_binomial_noiid.py', 'binomial'),
        ('run_poisson_noiid.py', 'poisson'),
        ('run_normal_noiid.py', 'normal'),
        ('run_exponential_noiid.py', 'exponential')
    ]
    
    print(f"\n📋 实验计划:")
    for i, (script, dist) in enumerate(experiments, 1):
        print(f"   {i}. {dist.upper()} 分布 ({script})")
    
    print(f"\n⚠️  预计总时间: ~{len(experiments)*15} 分钟")
    
    # 询问用户确认
    response = input(f"\n🤔 是否开始执行所有实验? (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是']:
        print("❌ 用户取消了实验")
        return
    
    # 记录总体开始时间
    total_start_time = time.time()
    successful_experiments = []
    failed_experiments = []
    
    print(f"\n🎬 开始批量实验!")
    print(f"🕐 总体开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 逐个运行实验
    for i, (script, dist) in enumerate(experiments, 1):
        print(f"\n🔄 实验进度: {i}/{len(experiments)}")
        
        success = run_experiment(script, dist)
        
        if success:
            successful_experiments.append(dist)
            print(f"✅ {dist.upper()} 实验记录已保存")
        else:
            failed_experiments.append(dist)
            
            # 询问是否继续
            if i < len(experiments):
                response = input(f"\n⚠️  {dist.upper()} 实验失败，是否继续下一个实验? (y/n): ").lower().strip()
                if response not in ['y', 'yes', '是']:
                    print(f"❌ 用户选择停止实验 (在 {dist.upper()} 失败后)")
                    break
        
        # 实验间短暂休息
        if i < len(experiments):
            print(f"\n⏸️ 休息 10 秒后继续下一个实验...")
            time.sleep(10)
    
    # 计算总耗时
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 生成最终报告
    print(f"\n{'='*80}")
    print("🎉 批量实验完成!")
    print(f"{'='*80}")
    print(f"⏱️ 总耗时: {total_duration/60:.1f} 分钟")
    print(f"✅ 成功实验: {len(successful_experiments)}")
    print(f"❌ 失败实验: {len(failed_experiments)}")
    
    if successful_experiments:
        print(f"\n🎯 成功完成的分布:")
        for dist in successful_experiments:
            print(f"   ✅ {dist.upper()}")
    
    if failed_experiments:
        print(f"\n💥 失败的分布:")
        for dist in failed_experiments:
            print(f"   ❌ {dist.upper()}")
    
    # 生成可视化
    if len(successful_experiments) >= 2:
        print(f"\n🎨 生成可视化结果...")
        try:
            result = subprocess.run([sys.executable, 'visualize_noiid.py'], 
                                  capture_output=True, 
                                  text=True, 
                                  cwd='.')
            if result.returncode == 0:
                print(f"📊 可视化图表生成成功!")
                print(f"📁 请查看 results/figures/ 目录")
            else:
                print(f"⚠️ 可视化生成失败: {result.stderr}")
        except Exception as e:
            print(f"⚠️ 可视化生成出错: {e}")
    
    # 提供后续建议
    print(f"\n💡 后续操作建议:")
    print(f"   📊 查看结果: python visualize_noiid.py")
    print(f"   📁 日志文件: results/logs/")
    print(f"   📈 图表文件: results/figures/")
    
    if len(successful_experiments) >= 2:
        print(f"\n🔍 Non-IID实验已完成，您现在可以:")
        print(f"   1. 比较不同分布的性能差异")
        print(f"   2. 分析Non-IID数据对各分布的影响")
        print(f"   3. 验证客户端选择策略的有效性")
    
    print(f"\n🎊 恭喜完成Non-IID联邦学习实验!")


if __name__ == '__main__':
    main()
