# experiments/run_iid_experiments.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from src.data.mnist_data import (
    load_mnist_data, 
    create_iid_partition,
    create_client_loaders,
    get_test_loader
)
from src.models.cnn_model import MNIST_CNN
from src.client_selection.uniform_selector import UniformSelector
from src.algorithms.fedavg import FederatedAveraging


def run_iid_baseline():
    """运行IID基准实验"""
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'base_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("="*80)
    print("实验：MNIST IID - FedAvg Baseline (均匀分布)")
    print("="*80)
    
    # 加载数据
    print("\n1. 加载数据...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', config['data']['data_dir'])
    train_dataset, test_dataset = load_mnist_data(data_dir)
    
    # 创建IID划分
    print("2. 创建IID数据划分...")
    client_indices = create_iid_partition(
        train_dataset, 
        num_clients=config['data']['num_clients'],
        seed=seed
    )
    
    # 创建客户端加载器
    client_loaders = create_client_loaders(
        train_dataset,
        client_indices,
        batch_size=config['federated']['batch_size']
    )
    
    # 创建测试集加载器
    test_loader = get_test_loader(test_dataset)
    
    # 创建模型
    print("3. 创建模型...")
    model = MNIST_CNN()
    print(f"   模型参数量: {model.count_parameters():,}")
    
    # 创建客户端选择器
    print("4. 创建客户端选择器...")
    selector = UniformSelector(
        num_clients=config['data']['num_clients'],
        config=config['client_selection']['config']
    )
    
    # 创建FedAvg算法
    print("5. 初始化FedAvg...")
    fed_config = {
        'clients_per_round': config['federated']['clients_per_round'],
        'local_epochs': config['federated']['local_epochs'],
        'learning_rate': config['training']['learning_rate'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    fed_alg = FederatedAveraging(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        client_selector=selector,
        config=fed_config
    )
    
    # 训练
    print("6. 开始训练...")
    history, selection_stats = fed_alg.train(
        num_rounds=config['training']['num_rounds'],
        eval_every=config['training']['eval_every'],
        target_accuracy=config['training']['target_accuracy']
    )
    
    # 保存结果
    print("7. 保存结果...")
    results = {
        'config': config,
        'history': history,
        'selection_stats': selection_stats
    }
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    torch.save(results, os.path.join(results_dir, 'iid_baseline_results.pt'))
    torch.save(model.state_dict(), os.path.join(models_dir, 'iid_baseline_model.pt'))
    
    print("\n实验完成！")
    print(f"结果已保存至: {os.path.join(results_dir, 'iid_baseline_results.pt')}")
    
    return results


if __name__ == '__main__':
    results = run_iid_baseline()
