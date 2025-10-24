#!/usr/bin/env python3
"""
Exponential分布客户端选择 - Non-IID版本
使用不同随机种子和Non-IID数据分布
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import logging
import yaml
import numpy as np

# 导入我们的模块
from src.data.mnist_data_noiid import create_noiid_loaders
from src.models.cnn_model import CNN
from src.client_selection.exponential_selector import ExponentialSelector
from src.algorithms.fedavg import FedAvgTrainer


def setup_logging(log_file):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config():
    """加载配置文件"""
    with open('configs/base_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Exponential分布专用种子
    SEED = 56789  # 与其他分布不同的种子
    set_seed(SEED)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/logs/exponential_noiid_{timestamp}.log"
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("开始 EXPONENTIAL 分布实验 (Non-IID版本)")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 80)
    
    # 加载配置
    config = load_config()
    
    # Exponential分布参数
    exponential_params = {
        'lambda': 1.5,
        'exp_mode': 'decay'
    }
    
    logger.info("🎯 Non-IID实验：EXPONENTIAL")
    logger.info(f"📊 分布类型: exponential")
    logger.info(f"⚙️  分布参数: {exponential_params}")
    logger.info(f"🎲 随机种子: {SEED}")
    logger.info(f"🔧 基础配置: 客户端数={config['num_clients']}, 每轮选择={config['clients_per_round']}, 轮数={config['num_rounds']}")
    logger.info("-" * 80)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  使用设备: {device}")
    
    logger.info(f"🎲 设置随机种子: {SEED}")
    
    # 1. 加载Non-IID MNIST数据
    logger.info("1️⃣ 加载Non-IID MNIST数据...")
    client_loaders, test_loader, data_manager = create_noiid_loaders(
        num_clients=config['num_clients'],
        alpha=0.3,  # 更强的Non-IID程度
        batch_size=config['batch_size'],
        data_dir='./data',
        min_samples=50
    )
    
    logger.info(f"   Non-IID数据划分为 {config['num_clients']} 个客户端")
    logger.info(f"   客户端加载器创建完成，批大小: {config['batch_size']}")
    
    # 打印前5个客户端的数据分布信息
    logger.info("📊 前5个客户端数据分布:")
    for client_id in range(5):
        info = data_manager.get_client_data_info(client_id)
        dominant = info['dominant_classes'][:3]
        logger.info(f"   客户端{client_id}: {info['total_samples']}样本, 主要类别: {dominant}")
    
    # 2. 创建CNN模型
    logger.info("2️⃣ 创建CNN模型...")
    model = CNN().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   模型参数量: {total_params:,}")
    
    # 3. 创建客户端选择器
    logger.info("3️⃣ 创建客户端选择器...")
    selector = ExponentialSelector(
        num_clients=config['num_clients'],
        clients_per_round=config['clients_per_round'],
        **exponential_params
    )
    logger.info(f"   选择器类型: Exponential")
    logger.info(f"   选择器配置: {exponential_params}")
    
    # 4. 创建联邦学习训练器
    logger.info("4️⃣ 创建联邦学习训练器...")
    trainer = FedAvgTrainer(
        model=model,
        device=device,
        learning_rate=config['learning_rate'],
        local_epochs=config['local_epochs']
    )
    
    logger.info("5️⃣ 开始联邦学习训练...")
    logger.info(f"   目标轮数: {config['num_rounds']}")
    logger.info("=" * 80)
    
    # 训练循环
    for round_num in range(1, config['num_rounds'] + 1):
        # 选择客户端
        selected_clients = selector.select_clients(round_num)
        
        # 收集选中客户端的数据加载器
        round_loaders = {cid: client_loaders[cid] for cid in selected_clients}
        
        # 执行一轮训练
        train_loss = trainer.train_round(round_loaders)
        
        # 评估模型
        test_accuracy, test_loss = trainer.evaluate(test_loader)
        
        # 记录结果
        selected_preview = selected_clients[:5] + (['...'] if len(selected_clients) > 5 else [])
        logger.info(f"🔄 Round {round_num:3d} | Acc: {test_accuracy:.4f} | Loss: {test_loss:.4f} | 选中: {selected_preview}")
        
        # 每50轮详细报告
        if round_num % 50 == 0:
            logger.info(f"📈 轮次 {round_num} 完成 - 测试准确率: {test_accuracy:.4f}")
    
    # 6. 保存结果
    logger.info("6️⃣ 保存训练结果...")
    
    # 保存模型
    model_path = f"results/models/exponential_noiid_model.pt"
    torch.save(trainer.model.state_dict(), model_path)
    logger.info(f"   模型已保存: {model_path}")
    
    # 保存训练历史
    results_path = f"results/exponential_noiid_results.pt"
    results = {
        'train_losses': trainer.train_losses,
        'test_accuracies': trainer.test_accuracies,
        'test_losses': trainer.test_losses,
        'config': config,
        'distribution': 'exponential',
        'distribution_params': exponential_params,
        'seed': SEED,
        'timestamp': timestamp,
        'noiid': True,
        'final_accuracy': trainer.test_accuracies[-1] if trainer.test_accuracies else 0
    }
    torch.save(results, results_path)
    logger.info(f"   训练历史已保存: {results_path}")
    
    # 最终报告
    final_accuracy = trainer.test_accuracies[-1] if trainer.test_accuracies else 0
    logger.info("=" * 80)
    logger.info("🎉 EXPONENTIAL分布实验完成 (Non-IID版本)")
    logger.info(f"📊 最终测试准确率: {final_accuracy:.4f}")
    logger.info(f"🎲 使用随机种子: {SEED}")
    logger.info(f"📁 结果保存在: {results_path}")
    logger.info(f"📋 日志保存在: {log_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
