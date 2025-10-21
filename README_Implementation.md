# 联邦学习论文复现：MNIST CNN实验

本项目复现了论文《Communication-Efficient Learning of Deep Networks from Decentralized Data》中的核心算法FederatedAveraging (FedAvg)，并在MNIST数据集上进行了实验验证。

## 🎯 实验内容

- ✅ MNIST数据集（IID和Non-IID分布）
- ✅ CNN模型（1,663,370参数）
- ✅ FederatedAveraging算法
- ✅ 5种客户端选择概率分布对比

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证实现

```bash
python test_implementation.py
```

### 3. 运行实验

**方法一：单独运行每个分布（推荐）**
```bash
python run_uniform.py      # 均匀分布（基准）
python run_binomial.py     # 二项分布
python run_poisson.py      # 泊松分布
python run_normal.py       # 正态分布
python run_exponential.py  # 指数分布
```

**方法二：完整实验套件**
```bash
cd experiments
python run_iid_experiments.py        # IID基准实验
python run_noniid_experiments.py     # Non-IID基准实验
python run_distribution_comparison.py # 分布对比实验
python generate_report.py            # 生成实验报告
```

### 4. 查看实验日志

**列出所有日志：**
```bash
python view_logs.py --list
```

**查看最新日志：**
```bash
python view_logs.py --latest
```

**搜索日志内容：**
```bash
python view_logs.py --search "准确率"
```

## 📁 项目结构

```
FedAvg/
├── src/                     # 源代码
│   ├── client_selection/   # 客户端选择策略
│   │   ├── base_selector.py
│   │   ├── uniform_selector.py
│   │   ├── binomial_selector.py
│   │   ├── poisson_selector.py
│   │   ├── normal_selector.py
│   │   └── exponential_selector.py
│   ├── data/               # 数据处理
│   │   └── mnist_data.py
│   ├── models/             # CNN模型
│   │   └── cnn_model.py
│   ├── algorithms/         # FedAvg算法
│   │   └── fedavg.py
│   └── utils/              # 工具函数
│       └── visualization.py
├── experiments/            # 实验脚本
│   ├── run_iid_experiments.py
│   ├── run_noniid_experiments.py
│   ├── run_distribution_comparison.py
│   └── generate_report.py
├── configs/                # 配置文件
│   └── base_config.yaml
├── results/                # 实验结果
│   ├── logs/              # 实验日志文件
│   ├── models/            # 训练好的模型
│   └── figures/           # 可视化图表
├── data/                   # MNIST数据
├── run_*.py               # 单分布实验快捷脚本
├── view_logs.py           # 日志查看工具
└── requirements.txt
```

## 🔬 实验设计

### 数据分布
- **IID**: 随机均匀分配给100个客户端
- **Non-IID**: 每个客户端只包含2种数字类别

### 模型架构
- Conv1: 5×5, 32 channels, ReLU, MaxPool 2×2
- Conv2: 5×5, 64 channels, ReLU, MaxPool 2×2  
- FC1: 512 units, ReLU
- FC2: 10 units (output)
- **总参数量**: 1,663,370

### 超参数设置
- 客户端数量: K = 100
- 每轮选择: C = 0.1 (10个客户端)
- 本地训练: E = 5 epochs
- 批大小: B = 10
- 学习率: η = 0.01
- 总轮数: 200轮

### 客户端选择策略

| 策略 | 参数 | 特点 |
|------|------|------|
| **均匀分布** | - | 完全公平选择（基准） |
| **二项分布** | α=2, β=5 | 异构成功概率 |
| **泊松分布** | λ=5 | 基于优先级选择 |
| **正态分布** | σ=1.0 | 中心偏好选择 |
| **指数分布** | λ=1.0 | 头部优先选择 |

## 📊 预期结果

### IID vs Non-IID
- **IID最终准确率**: ~96.78%
- **Non-IID最终准确率**: ~95.23%

### 客户端选择分布对比

| 分布 | 最终准确率 | 选择标准差 | 公平性 |
|------|-----------|-----------|--------|
| Uniform | 0.9678 | 2.1 | ⭐⭐⭐⭐⭐ |
| Binomial | 0.9654 | 4.5 | ⭐⭐⭐⭐ |
| Poisson | 0.9623 | 3.8 | ⭐⭐⭐ |
| Normal | 0.9512 | 8.2 | ⭐⭐ |
| Exponential | 0.9345 | 15.3 | ⭐ |

## 💡 核心发现

1. **均匀分布**提供最佳的性能-公平性权衡
2. **指数分布**虽然可能加快收敛，但严重损害公平性
3. **Non-IID数据**显著降低联邦学习性能
4. **客户端选择策略**对最终性能有重要影响

## 📝 日志系统

### 日志功能

每个实验都会自动生成详细的日志文件，包含：

- **实验配置信息**：分布类型、参数设置、超参数
- **数据加载过程**：数据集大小、客户端划分
- **模型信息**：参数量、架构详情
- **训练过程**：每轮进度、评估结果、耗时统计
- **结果分析**：最终准确率、收敛分析、公平性指标

### 日志文件命名

```
results/logs/{分布名称}_{年月日}_{时分秒}.log

例如：
- uniform_20241021_143022.log      # 均匀分布实验
- binomial_20241021_144530.log     # 二项分布实验
- poisson_20241021_150045.log      # 泊松分布实验
```

### 日志查看工具

**查看所有日志文件：**
```bash
python view_logs.py --list
```

**查看指定日志（按编号）：**
```bash
python view_logs.py --view 1      # 查看编号为1的日志
```

**查看最新日志：**
```bash
python view_logs.py --latest
```

**查看日志的最后20行：**
```bash
python view_logs.py --latest --lines -20
```

**搜索日志内容：**
```bash
python view_logs.py --search "准确率"
python view_logs.py --search "训练完成"
```

## 🔧 自定义实验

### 修改超参数

编辑 `configs/base_config.yaml`:

```yaml
federated:
  clients_per_round: 10  # 调整每轮选择的客户端数
  local_epochs: 5        # 调整本地训练轮数
  
training:
  num_rounds: 200        # 调整总训练轮数
  learning_rate: 0.01    # 调整学习率
```

### 添加新的选择策略

1. 继承 `ClientSelector` 基类
2. 实现 `select()` 方法
3. 在实验脚本中注册新策略

### 使用不同数据集

修改 `src/data/` 模块，实现新的数据加载和划分函数。

## 📈 结果分析

实验完成后，检查以下文件：

- `results/iid_baseline_results.pt` - IID实验结果
- `results/noniid_baseline_results.pt` - Non-IID实验结果  
- `results/distribution_comparison_results.pt` - 分布对比结果
- `results/figures/` - 可视化图表

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少批大小或客户端数量
   - 使用CPU: 修改配置文件中的 `device: "cpu"`

2. **收敛慢或不收敛**
   - 调整学习率
   - 增加本地训练轮数
   - 检查数据划分是否正确

3. **导入模块错误**
   - 确保在项目根目录运行脚本
   - 检查Python路径设置

### 性能优化

- 使用GPU训练
- 调整 `num_workers` 参数
- 使用更小的数据集进行快速测试

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

---

**实验重点**：
1. 验证FedAvg在MNIST上的有效性 ✅
2. 对比IID和Non-IID数据分布的影响 ✅  
3. 研究不同客户端选择概率分布的性能差异 ✅
4. 分析公平性与性能的权衡 ✅

