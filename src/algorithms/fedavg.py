# src/algorithms/fedavg.py

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


class FederatedAveraging:
    """
    联邦平均算法（FedAvg）
    
    参考论文Algorithm 1（第5页）
    """
    
    def __init__(self,
                 model: nn.Module,
                 client_loaders: List[torch.utils.data.DataLoader],
                 test_loader: torch.utils.data.DataLoader,
                 client_selector,
                 config: Dict[str, Any]):
        """
        初始化FedAvg算法
        
        Args:
            model: 全局模型
            client_loaders: 客户端数据加载器列表
            test_loader: 测试集加载器
            client_selector: 客户端选择器
            config: 配置字典
        """
        self.global_model = model
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.client_selector = client_selector
        self.config = config
        
        # 超参数
        self.num_clients = len(client_loaders)
        self.clients_per_round = config['clients_per_round']
        self.local_epochs = config['local_epochs']  # E
        self.learning_rate = config['learning_rate']  # η
        self.device = config['device']
        
        # 移动模型到设备
        self.global_model.to(self.device)
        
        # 训练历史
        self.history = {
            'rounds': [],
            'train_loss': [],
            'test_acc': [],
            'test_loss': []
        }
        
        print(f"FedAvg初始化完成")
        print(f"  - 客户端总数: {self.num_clients}")
        print(f"  - 每轮选择: {self.clients_per_round}")
        print(f"  - 本地epoch: {self.local_epochs}")
        print(f"  - 学习率: {self.learning_rate}")
        print(f"  - 设备: {self.device}")
    
    def client_update(self, 
                      client_id: int, 
                      global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        客户端本地更新（Algorithm 1中的ClientUpdate）
        
        Args:
            client_id: 客户端ID
            global_weights: 全局模型参数
            
        Returns:
            updated_weights: 更新后的模型参数
        """
        # 创建本地模型副本
        local_model = copy.deepcopy(self.global_model)
        local_model.load_state_dict(global_weights)
        local_model.train()
        
        # 本地优化器（使用SGD，无momentum）
        optimizer = optim.SGD(local_model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 本地训练E个epoch
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.client_loaders[client_id]):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        # 返回更新后的模型参数
        return local_model.state_dict()
    
    def aggregate_models(self, 
                        client_weights_list: List[Dict[str, torch.Tensor]], 
                        client_data_sizes: List[int]) -> None:
        """
        聚合客户端模型（加权平均）
        
        Args:
            client_weights_list: 客户端模型参数列表
            client_data_sizes: 客户端数据量列表
        """
        # 计算总数据量
        total_size = sum(client_data_sizes)
        
        # 加权平均
        global_weights = self.global_model.state_dict()
        
        # 初始化为零
        for key in global_weights.keys():
            global_weights[key] = torch.zeros_like(global_weights[key])
        
        # 加权求和
        for client_weights, data_size in zip(client_weights_list, client_data_sizes):
            weight = data_size / total_size
            for key in global_weights.keys():
                global_weights[key] += client_weights[key] * weight
        
        # 更新全局模型
        self.global_model.load_state_dict(global_weights)
    
    def evaluate(self) -> Tuple[float, float]:
        """
        评估全局模型
        
        Returns:
            accuracy: 测试准确率
            loss: 测试损失
        """
        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                
                # 累计损失
                test_loss += criterion(output, target).item()
                
                # 统计准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= len(self.test_loader)
        accuracy = correct / total
        
        return accuracy, test_loss
    
    def train_round(self, round_num: int) -> Dict[str, Any]:
        """
        执行一轮联邦学习
        
        Args:
            round_num: 当前轮次
            
        Returns:
            round_info: 本轮详细信息
        """
        # 使用客户端选择器选择客户端
        selected_clients = self.client_selector.select(
            num_select=self.clients_per_round,
            round_num=round_num
        )
        
        # 获取当前全局模型参数
        global_weights = self.global_model.state_dict()
        
        # 客户端本地更新
        client_weights_list = []
        client_data_sizes = []
        
        for client_id in selected_clients:
            # 客户端更新
            updated_weights = self.client_update(client_id, global_weights)
            client_weights_list.append(updated_weights)
            
            # 记录客户端数据量（用于加权平均）
            data_size = len(self.client_loaders[client_id].dataset)
            client_data_sizes.append(data_size)
        
        # 聚合模型
        self.aggregate_models(client_weights_list, client_data_sizes)
        
        # 记录选择历史
        self.client_selector.record_selection(selected_clients, round_num)
        
        # 返回本轮信息
        return {
            'round': round_num,
            'selected_clients': selected_clients.tolist(),
            'num_selected': len(selected_clients),
            'client_data_sizes': client_data_sizes
        }
    
    def train(self, 
              num_rounds: int, 
              eval_every: int = 10,
              target_accuracy: float = None,
              logger=None) -> Tuple[Dict, Dict]:
        """
        完整训练流程
        
        Args:
            num_rounds: 总训练轮数
            eval_every: 每隔多少轮评估一次
            target_accuracy: 目标准确率（达到后可提前停止）
            logger: 日志记录器
            
        Returns:
            history: 训练历史
            selection_stats: 客户端选择统计
        """
        # 扩展历史记录结构，记录每轮详细数据
        detailed_history = {
            'rounds': [],
            'test_acc': [],
            'test_loss': [],
            'selected_clients': [],
            'round_details': []
        }
        
        log_func = logger.info if logger else print
        
        log_func(f"\n{'='*80}")
        log_func(f"开始训练")
        log_func(f"  - 总轮数: {num_rounds}")
        log_func(f"  - 客户端选择器: {self.client_selector.get_name()}")
        log_func(f"{'='*80}\n")
        
        # 初始评估
        init_acc, init_loss = self.evaluate()
        log_func(f"📊 初始性能 | Test Acc: {init_acc:.4f} | Test Loss: {init_loss:.4f}")
        log_func("-"*60)
        
        # 训练循环
        for round_num in tqdm(range(1, num_rounds + 1), desc="Training"):
            # 训练一轮
            round_info = self.train_round(round_num)
            
            # 每轮都进行评估以获得完整数据
            acc, loss = self.evaluate()
            
            # 记录详细历史
            detailed_history['rounds'].append(round_num)
            detailed_history['test_acc'].append(acc)
            detailed_history['test_loss'].append(loss)
            detailed_history['selected_clients'].append(round_info['selected_clients'])
            detailed_history['round_details'].append(round_info)
            
            # 每轮都记录基本信息到日志
            selected_clients_str = str(round_info['selected_clients'][:5])  # 只显示前5个
            if len(round_info['selected_clients']) > 5:
                selected_clients_str = selected_clients_str[:-1] + ", ...]"
            
            log_func(f"🔄 Round {round_num:3d} | Acc: {acc:.4f} | Loss: {loss:.4f} | "
                    f"选中: {selected_clients_str}")
            
            # 定期显示详细信息
            if round_num % eval_every == 0 or round_num == 1:
                log_func(f"   📈 详细 Round {round_num:3d} | Test Acc: {acc:.4f} | Test Loss: {loss:.4f}")
                log_func(f"   👥 选中客户端: {round_info['selected_clients']}")
                log_func(f"   📊 选中数量: {len(round_info['selected_clients'])}/{self.num_clients}")
                log_func("-"*60)
                
                # 检查是否达到目标准确率
                if target_accuracy is not None and acc >= target_accuracy:
                    log_func(f"\n🎯 达到目标准确率 {target_accuracy:.2%}，提前停止训练")
                    break
        
        # 最终评估
        final_acc, final_loss = self.evaluate()
        log_func(f"\n{'='*80}")
        log_func(f"🎉 训练完成")
        log_func(f"  - 最终准确率: {final_acc:.4f}")
        log_func(f"  - 最终损失: {final_loss:.4f}")
        
        # 显示训练过程统计
        if len(detailed_history['test_acc']) > 1:
            max_acc = max(detailed_history['test_acc'])
            min_loss = min(detailed_history['test_loss'])
            improvement = detailed_history['test_acc'][-1] - detailed_history['test_acc'][0]
            log_func(f"  - 最佳准确率: {max_acc:.4f}")
            log_func(f"  - 最低损失: {min_loss:.4f}")
            log_func(f"  - 准确率提升: {improvement:+.4f}")
        
        log_func(f"{'='*80}\n")
        
        # 获取客户端选择统计
        selection_stats = self.client_selector.get_selection_statistics()
        
        # 更新self.history为详细历史
        self.history = detailed_history
        
        return detailed_history, selection_stats
