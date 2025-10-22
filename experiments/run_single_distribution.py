# experiments/run_single_distribution.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
import argparse
import logging
import time
from datetime import datetime
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


def setup_logging(distribution_name, log_dir='../results/logs'):
    """设置日志记录"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{distribution_name}_{timestamp}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"="*80)
    logger.info(f"开始 {distribution_name.upper()} 分布实验")
    logger.info(f"日志文件: {log_filename}")
    logger.info(f"="*80)
    
    return logger, log_filename


def get_distribution_config(distribution_name):
    """获取各分布的默认配置"""
    configs = {
        'uniform': {},
        'binomial': {'alpha': 2, 'beta': 5, 'static': True},
        'poisson': {'lambda': 5, 'static': True},
        'normal': {'sigma': 1.0, 'weight_method': 'spatial'},
        'exponential': {'lambda': 1.0, 'exp_mode': 'decay'}
    }
    return configs.get(distribution_name, {})


def run_single_distribution(distribution_name, custom_config=None):
    """运行单个分布的实验"""
    
    # 设置日志记录
    logger, log_filename = setup_logging(distribution_name)
    start_time = time.time()
    
    logger.info(f"🎯 单分布实验：{distribution_name.upper()}")
    
    # 加载基础配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'base_config.yaml')
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 获取分布配置
    if custom_config:
        selector_config = custom_config
    else:
        selector_config = get_distribution_config(distribution_name)
    
    logger.info(f"📊 分布类型: {distribution_name}")
    logger.info(f"⚙️  分布参数: {selector_config}")
    logger.info(f"🔧 基础配置: 客户端数={base_config['data']['num_clients']}, 每轮选择={base_config['federated']['clients_per_round']}, 轮数={base_config['training']['num_rounds']}")
    logger.info("-"*80)
    
    # 设置随机种子
    seed = base_config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"🎲 设置随机种子: {seed}")
    
    # 加载数据
    logger.info("1️⃣ 加载MNIST数据...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', base_config['data']['data_dir'])
    train_dataset, test_dataset = load_mnist_data(data_dir)
    test_loader = get_test_loader(test_dataset)
    logger.info(f"   训练集大小: {len(train_dataset):,}, 测试集大小: {len(test_dataset):,}")
    
    # 创建IID划分
    logger.info("2️⃣ 创建IID数据划分...")
    client_indices = create_iid_partition(
        train_dataset,
        num_clients=base_config['data']['num_clients'],
        seed=seed
    )
    logger.info(f"   划分为 {len(client_indices)} 个客户端，每客户端约 {len(client_indices[0])} 个样本")
    
    # 创建客户端加载器
    client_loaders = create_client_loaders(
        train_dataset,
        client_indices,
        batch_size=base_config['federated']['batch_size']
    )
    logger.info(f"   客户端加载器创建完成，批大小: {base_config['federated']['batch_size']}")
    
    # 创建模型
    logger.info("3️⃣ 创建CNN模型...")
    model = MNIST_CNN()
    param_count = model.count_parameters()
    logger.info(f"   模型参数量: {param_count:,}")
    
    # 创建选择器
    logger.info("4️⃣ 创建客户端选择器...")
    selector_classes = {
        'uniform': UniformSelector,
        'binomial': BinomialSelector,
        'poisson': PoissonSelector,
        'normal': NormalSelector,
        'exponential': ExponentialSelector
    }
    
    if distribution_name not in selector_classes:
        raise ValueError(f"未知的分布类型: {distribution_name}。支持的类型: {list(selector_classes.keys())}")
    
    selector = selector_classes[distribution_name](
        num_clients=base_config['data']['num_clients'],
        config=selector_config
    )
    
    logger.info(f"   选择器类型: {selector.get_name()}")
    logger.info(f"   选择器配置: {selector_config}")
    
    # 创建FedAvg
    logger.info("5️⃣ 初始化FederatedAveraging...")
    fed_config = {
        'clients_per_round': base_config['federated']['clients_per_round'],
        'local_epochs': base_config['federated']['local_epochs'],
        'learning_rate': base_config['training']['learning_rate'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"   训练设备: {fed_config['device']}")
    logger.info(f"   每轮选择客户端数: {fed_config['clients_per_round']}")
    logger.info(f"   本地训练轮数: {fed_config['local_epochs']}")
    logger.info(f"   学习率: {fed_config['learning_rate']}")
    
    fed_alg = FederatedAveraging(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        client_selector=selector,
        config=fed_config
    )
    
    # 训练
    logger.info("6️⃣ 开始训练...")
    train_start_time = time.time()
    history, selection_stats = fed_alg.train(
        num_rounds=base_config['training']['num_rounds'],
        eval_every=base_config['training']['eval_every'],
        logger=logger
    )
    train_end_time = time.time()
    training_duration = train_end_time - train_start_time
    logger.info(f"✅ 训练完成，耗时: {training_duration:.1f} 秒 ({training_duration/60:.1f} 分钟)")
    
    # 保存结果
    logger.info("7️⃣ 保存结果...")
    
    # 计算总耗时
    total_duration = time.time() - start_time
    
    results = {
        'distribution': distribution_name,
        'config': selector_config,
        'base_config': base_config,
        'history': history,
        'selection_stats': selection_stats,
        'log_filename': log_filename,
        'training_duration': training_duration,
        'total_duration': total_duration,
        'timestamp': datetime.now().isoformat()
    }
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 保存文件
    result_file = os.path.join(results_dir, f'{distribution_name}_results.pt')
    model_file = os.path.join(models_dir, f'{distribution_name}_model.pt')
    
    torch.save(results, result_file)
    torch.save(model.state_dict(), model_file)
    
    logger.info(f"   结果文件: {result_file}")
    logger.info(f"   模型文件: {model_file}")
    
    # 记录详细结果摘要
    logger.info("="*80)
    logger.info("📊 实验结果摘要")
    logger.info("="*80)
    logger.info(f"🎯 分布类型: {distribution_name}")
    logger.info(f"📈 最终准确率: {history['test_acc'][-1]:.4f}")
    logger.info(f"📉 最终损失: {history['test_loss'][-1]:.4f}")
    
    # 收敛分析
    if len(history['test_acc']) > 1:
        initial_acc = history['test_acc'][0] if len(history['test_acc']) > 0 else 0
        final_acc = history['test_acc'][-1]
        improvement = final_acc - initial_acc
        logger.info(f"🚀 准确率提升: {improvement:.4f} ({initial_acc:.4f} → {final_acc:.4f})")
    
    # 客户端选择统计
    logger.info(f"👥 客户端选择统计:")
    logger.info(f"   - 平均选择次数: {selection_stats['mean']:.1f}")
    logger.info(f"   - 选择标准差: {selection_stats['std']:.1f}")
    logger.info(f"   - 最大选择次数: {selection_stats['max']}")
    logger.info(f"   - 最小选择次数: {selection_stats['min']}")
    fairness_ratio = selection_stats['max'] / max(selection_stats['min'], 1)
    logger.info(f"   - 公平性比率 (越接近1越公平): {fairness_ratio:.2f}")
    
    # 时间统计
    logger.info(f"⏱️ 时间统计:")
    logger.info(f"   - 训练耗时: {training_duration:.1f} 秒 ({training_duration/60:.1f} 分钟)")
    logger.info(f"   - 总耗时: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟)")
    logger.info(f"   - 平均每轮耗时: {training_duration/base_config['training']['num_rounds']:.2f} 秒")
    
    logger.info("="*80)
    logger.info(f"🎉 {distribution_name.upper()} 分布实验成功完成！")
    logger.info(f"📝 完整日志已保存至: {log_filename}")
    logger.info("="*80)
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行单个分布的联邦学习实验')
    parser.add_argument('distribution', 
                       choices=['uniform', 'binomial', 'poisson', 'normal', 'exponential'],
                       help='要运行的分布类型')
    parser.add_argument('--rounds', type=int, default=None, 
                       help='训练轮数 (默认使用配置文件中的值)')
    parser.add_argument('--clients', type=int, default=None,
                       help='每轮选择的客户端数 (默认使用配置文件中的值)')
    
    args = parser.parse_args()
    
    # 自定义配置（如果有的话）
    custom_config = get_distribution_config(args.distribution)
    
    try:
        results = run_single_distribution(args.distribution, custom_config)
        print("\n🎉 实验成功完成！")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
