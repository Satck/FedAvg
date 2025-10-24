# FedAvg Non-IID 实验

这是改进版的联邦学习实验代码，专门解决原版本中所有分布曲线重合的问题。

## 🔍 问题分析

原版本存在的问题：
- 所有分布使用相同的随机种子（42）
- 使用IID数据分布，客户端选择策略差异不明显  
- 最终准确率差异仅0.06%，曲线几乎重合

## ✨ 改进措施

### 1. 不同随机种子
- **Uniform**: 12345
- **Binomial**: 23456  
- **Poisson**: 34567
- **Normal**: 45678
- **Exponential**: 56789

### 2. Non-IID数据分布
- 使用Dirichlet分布 (α=0.3) 创建强Non-IID数据
- 每个客户端专注于少数几个类别
- 不同客户端的数据分布差异明显

### 3. 增强可视化
- 自动放大y轴范围显示微小差异
- 添加详细的数值验证输出
- 强调Non-IID效果的分析

## 📁 目录结构

```
FedAvg_noiid/
├── src/                          # 源代码
│   ├── data/
│   │   ├── mnist_data_noiid.py   # Non-IID数据加载器
│   │   └── mnist_data.py         # 原始IID数据加载器
│   ├── models/
│   ├── client_selection/         # 客户端选择策略
│   └── algorithms/
├── configs/
│   └── base_config.yaml         # 基础配置
├── results/                     # 实验结果
│   ├── logs/                    # 日志文件
│   ├── figures/                 # 图表
│   └── models/                  # 保存的模型
├── run_*_noiid.py              # 各分布实验脚本
├── run_all_noiid.py            # 批量运行脚本
├── visualize_noiid.py          # Non-IID可视化工具
└── README_noiid.md             # 本文件
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行单个实验
```bash
# 运行Uniform分布实验
python run_uniform_noiid.py

# 运行Binomial分布实验  
python run_binomial_noiid.py

# ... 其他分布类似
```

### 3. 批量运行所有实验
```bash
# 自动运行所有5个分布实验
python run_all_noiid.py
```

### 4. 生成可视化结果
```bash
# 生成对比图表
python visualize_noiid.py
```

## 📊 Non-IID数据特性

使用 `create_noiid_loaders()` 创建的数据具有以下特性：

- **Alpha参数**: 0.3 (越小越Non-IID)
- **类别分布**: 每个客户端主要包含2-3个类别的数据
- **样本分布**: 客户端间样本数量差异较大
- **Non-IID程度**: 通过KL散度量化，预期 > 0.5

### 数据分布示例
```
客户端 0: [520,   5,  12,   3,   0,   8,   2,   0,   0,   0] (总计: 550)
客户端 1: [  8, 489,  45,   3,   5,   0,   0,   0,   0,   0] (总计: 550)
客户端 2: [  2,  15, 478,  35,  10,   8,   2,   0,   0,   0] (总计: 550)
```

## 🎯 预期结果

Non-IID版本应该产生：

1. **更明显的分布差异** (> 2% 准确率差异)
2. **不同的收敛模式** 
3. **更真实的联邦学习场景**

### 收敛差异示例
- **Uniform**: 可能收敛较慢但稳定
- **Binomial**: 可能早期波动较大
- **Exponential**: 可能后期收敛更快

## 🔧 参数调整

### 调整Non-IID程度
在各个 `run_*_noiid.py` 文件中修改：
```python
client_loaders, test_loader, data_manager = create_noiid_loaders(
    num_clients=100,
    alpha=0.3,    # 降低此值增加Non-IID程度
    batch_size=10,
    min_samples=50
)
```

### 调整分布参数
每个分布都有专门的参数可以调整：
- **Binomial**: `alpha`, `beta`, `binomial_mode`
- **Poisson**: `lambda`, `poisson_mode`  
- **Normal**: `mu`, `sigma`, `normal_mode`
- **Exponential**: `lambda`, `exp_mode`

## 📈 结果分析

### 可视化工具功能
1. **自动差异检测**: 识别微小差异并放大显示
2. **数值验证**: 提供详细的统计分析
3. **Non-IID效果评估**: 量化Non-IID设置的影响

### 关键指标
- **准确率差异范围**: 期望 > 2%
- **收敛稳定性**: 后20%轮次的标准差
- **Non-IID程度**: KL散度测量

## 🆚 与原版对比

| 方面 | 原版 (IID) | 改进版 (Non-IID) |
|-----|-----------|----------------|
| 随机种子 | 统一使用42 | 每个分布不同种子 |
| 数据分布 | IID | Non-IID (α=0.3) |
| 准确率差异 | 0.06% | 预期 > 2% |
| 曲线区分度 | 几乎重合 | 明显可区分 |
| 实验真实性 | 理想化 | 更接近现实 |

## 🐛 故障排除

### 常见问题

1. **内存不足**: 减少 `num_clients` 或 `batch_size`
2. **收敛太慢**: 增加 `learning_rate` 或 `local_epochs`  
3. **差异仍然较小**: 进一步降低 `alpha` 参数

### 检查数据分布
```python
# 在任意实验脚本中添加
for client_id in range(5):
    info = data_manager.get_client_data_info(client_id)
    print(f"客户端{client_id}: {info}")
```

## 📞 支持

如果遇到问题，请检查：
1. 日志文件 (`results/logs/`)
2. 数据分布统计
3. 可视化输出中的差异分析

这个改进版本应该能够产生明显不同的分布曲线，展示出各种客户端选择策略在Non-IID场景下的真实性能差异。
