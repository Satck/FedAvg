# run_all_distributions.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
import subprocess
from pathlib import Path


def print_header():
    """打印标题"""
    print("\n" + "="*80)
    print("🚀 联邦学习客户端选择分布对比实验 - 批量运行")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("将依次运行以下5个分布的实验:")
    print("  1. Uniform (均匀分布)")
    print("  2. Binomial (二项分布)")
    print("  3. Poisson (泊松分布)")
    print("  4. Normal (正态分布)")
    print("  5. Exponential (指数分布)")
    print("="*80)


def run_single_experiment(distribution_name, experiment_index, total_experiments):
    """运行单个实验"""
    print(f"\n📋 实验 {experiment_index}/{total_experiments}: {distribution_name.upper()} 分布")
    print("-"*60)
    
    start_time = time.time()
    
    try:
        # 使用现有的便捷脚本
        script_path = f"run_{distribution_name}.py"
        
        if not os.path.exists(script_path):
            print(f"❌ 未找到脚本文件: {script_path}")
            return False
        
        print(f"🔄 执行: python {script_path}")
        
        # 运行实验
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {distribution_name.upper()} 分布实验成功完成")
            print(f"⏱️  耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
            
            # 检查结果文件是否生成
            result_file = f"results/{distribution_name}_results.pt"
            if os.path.exists(result_file):
                print(f"📁 结果文件已生成: {result_file}")
            else:
                print(f"⚠️  警告: 未找到结果文件 {result_file}")
            
            return True
        else:
            print(f"❌ {distribution_name.upper()} 分布实验失败")
            print(f"错误代码: {result.returncode}")
            if result.stderr:
                print(f"错误信息: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ {distribution_name.upper()} 分布实验异常: {e}")
        print(f"⏱️  耗时: {duration:.1f} 秒")
        return False


def check_prerequisites():
    """检查运行前提条件"""
    print("\n🔍 检查运行前提条件...")
    
    # 检查必需的脚本文件
    required_scripts = [
        "run_uniform.py",
        "run_binomial.py", 
        "run_poisson.py",
        "run_normal.py",
        "run_exponential.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ 缺少必需的脚本文件: {missing_scripts}")
        return False
    
    # 检查配置文件
    config_file = "configs/base_config.yaml"
    if not os.path.exists(config_file):
        print(f"❌ 缺少配置文件: {config_file}")
        return False
    
    # 创建必要的目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    print("✅ 前提条件检查通过")
    return True


def run_comparison_analysis():
    """运行对比分析"""
    print("\n🔍 开始对比分析...")
    print("-"*60)
    
    try:
        # 运行对比分析脚本
        analysis_script = "experiments/compare_distributions.py"
        
        if not os.path.exists(analysis_script):
            print(f"❌ 未找到对比分析脚本: {analysis_script}")
            return False
        
        print(f"🔄 执行: python {analysis_script}")
        
        result = subprocess.run(
            [sys.executable, analysis_script],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print("✅ 对比分析完成")
            if result.stdout:
                # 显示分析输出的最后几行
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-10:]:  # 显示最后10行
                    if line.strip():
                        print(f"   {line}")
            return True
        else:
            print("❌ 对比分析失败")
            if result.stderr:
                print(f"错误信息: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"❌ 对比分析异常: {e}")
        return False


def generate_summary(success_count, total_experiments, total_duration):
    """生成实验摘要"""
    print("\n" + "="*80)
    print("📊 实验摘要")
    print("="*80)
    print(f"✅ 成功完成实验: {success_count}/{total_experiments}")
    print(f"⏱️  总耗时: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟)")
    print(f"🕒 平均每个实验: {total_duration/total_experiments:.1f} 秒 ({total_duration/total_experiments/60:.1f} 分钟)")
    
    if success_count == total_experiments:
        print("🎉 所有实验都成功完成！")
    else:
        failed_count = total_experiments - success_count
        print(f"⚠️  有 {failed_count} 个实验失败")
    
    # 列出生成的文件
    print("\n📁 生成的文件:")
    results_dir = Path("results")
    
    # 结果文件
    for pt_file in results_dir.glob("*_results.pt"):
        print(f"   📊 {pt_file}")
    
    # 模型文件  
    models_dir = results_dir / "models"
    if models_dir.exists():
        for model_file in models_dir.glob("*.pt"):
            print(f"   🤖 {model_file}")
    
    # 日志文件
    logs_dir = results_dir / "logs"
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            print(f"   📝 {log_file}")
    
    # 图表文件
    figures_dir = results_dir / "figures"
    if figures_dir.exists():
        for fig_file in figures_dir.glob("*.png"):
            print(f"   📊 {fig_file}")
    
    # 报告文件
    for report_file in results_dir.glob("comparison_report_*.txt"):
        print(f"   📋 {report_file}")
    
    print("="*80)


def main():
    """主函数"""
    overall_start_time = time.time()
    
    print_header()
    
    # 检查前提条件
    if not check_prerequisites():
        print("\n❌ 前提条件检查失败，无法继续")
        return
    
    # 分布列表
    distributions = ['uniform', 'binomial', 'poisson', 'normal', 'exponential']
    total_experiments = len(distributions)
    success_count = 0
    
    # 依次运行每个实验
    for i, distribution in enumerate(distributions, 1):
        success = run_single_experiment(distribution, i, total_experiments)
        if success:
            success_count += 1
        
        # 在实验之间稍作休息
        if i < total_experiments:
            print(f"\n⏸️  休息 3 秒后继续下一个实验...")
            time.sleep(3)
    
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    
    # 生成实验摘要
    generate_summary(success_count, total_experiments, total_duration)
    
    # 如果所有实验都成功，运行对比分析
    if success_count == total_experiments:
        print(f"\n🎯 所有实验完成，开始生成对比分析...")
        analysis_success = run_comparison_analysis()
        
        if analysis_success:
            print("\n🎉 完整的实验流程已全部完成！")
            print("📊 您可以在 results/figures/ 目录中查看生成的对比图表")
            print("📋 详细报告保存在 results/ 目录中")
        else:
            print("\n⚠️  实验完成但对比分析失败")
            print("您可以手动运行: python experiments/compare_distributions.py")
    else:
        print(f"\n⚠️  由于有实验失败，跳过对比分析")
        print("请检查失败的实验并重新运行")
    
    print(f"\n🏁 批量运行结束，总耗时: {total_duration/60:.1f} 分钟")


if __name__ == '__main__':
    main()
