# src/models/cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """
    MNIST CNN模型
    
    架构（论文第5页）：
    - Conv1: 5x5, 32 channels, ReLU, MaxPool 2x2
    - Conv2: 5x5, 64 channels, ReLU, MaxPool 2x2
    - FC1: 2048 units, ReLU  # 修改为更大的隐藏层
    - FC2: 10 units (output)
    
    总参数量: ~1,663,370
    """
    
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=0)
        
        # 全连接层
        # 输入尺寸计算: 28 -> 24 -> 12 -> 8 -> 4，所以是 4*4*64 = 1024
        # 修改为更大的隐藏层以达到论文要求的参数量
        self.fc1 = nn.Linear(4 * 4 * 64, 1600)  # 增大隐藏层
        self.fc2 = nn.Linear(1600, 10)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，shape (batch_size, 1, 28, 28)
            
        Returns:
            output: 输出张量，shape (batch_size, 10)
        """
        # 第一层卷积 + 池化: 28x28 -> 24x24 -> 12x12
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 第二层卷积 + 池化: 12x12 -> 8x8 -> 4x4
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 展平: (batch, 64, 4, 4) -> (batch, 1024)
        x = x.view(-1, 4 * 4 * 64)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model():
    """创建模型实例"""
    model = MNIST_CNN()
    return model


def test_model():
    """测试模型"""
    model = MNIST_CNN()
    
    # 验证参数量
    param_count = model.count_parameters()
    print(f"模型参数量: {param_count:,} (应为 1,663,370)")
    
    # 验证前向传播
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    print(f"输入shape: {x.shape}")
    print(f"输出shape: {output.shape} (应为 torch.Size([{batch_size}, 10]))")
    
    # 验证梯度传播
    loss = output.sum()
    loss.backward()
    print("梯度反向传播成功")
    
    # 打印模型结构
    print("\n模型结构:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    
    return model


if __name__ == '__main__':
    test_model()
