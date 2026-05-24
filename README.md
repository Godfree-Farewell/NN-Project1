# 神经网络与深度学习 - 项目一

**课程：** 神经网络与深度学习（Neural Network and Deep Learning）
**姓名：** 龙政宇
**学号：** 22307130289
**日期：** 2026年5月22日

---

## 项目概述

本项目使用 NumPy 从零实现了一套自定义神经网络库，完成以下内容：

- **Part A：** MLP（多层感知机）基线模型，用于 MNIST 手写数字分类
- **Part B：** CNN（卷积神经网络），手动实现 conv2D 算子
- **Part C：** 两个额外探索方向
  - 方向一：优化器（动量 Momentum）
  - 方向五：错误分析与可视化

---

## 仓库结构

```
NN_Project_Solution/
├── codes/
│   ├── mynn/                      # 自定义神经网络库
│   │   ├── __init__.py
│   │   ├── op.py                  # 核心算子（Linear、Conv2D、ReLU 等）
│   │   ├── models.py              # 模型定义（MLP、CNN）
│   │   ├── optimizer.py           # 优化器（SGD、Momentum）
│   │   ├── lr_scheduler.py        # 学习率调度器
│   │   ├── runner.py              # 训练运行器
│   │   └── metric.py              # 评估指标
│   ├── draw_tools/                # 可视化工具
│   │   ├── __init__.py
│   │   └── plot.py
│   ├── dataset/MNIST/             # MNIST 数据集
│   │   ├── train-images-idx3-ubyte.gz
│   │   ├── train-labels-idx1-ubyte.gz
│   │   ├── t10k-images-idx3-ubyte.gz
│   │   ├── t10k-labels-idx1-ubyte.gz
│   │   └── README.md
│   ├── test_train.py              # 训练脚本
│   ├── test_model.py              # 测试脚本
│   ├── additional_experiments.py  # Part C 额外实验
│   ├── test_basic.py              # 基本功能测试
│   └── README.md                  # 代码说明文档
├── report/
│   └── project_report.md          # 项目报告
└── README.md                      # 本文件
```

---

## 环境依赖

- Python 3.8+
- NumPy
- Matplotlib

安装依赖：
```bash
pip install numpy matplotlib
```

---

## 快速开始

### 1. 基本功能测试

验证所有组件是否正常工作：
```bash
cd codes
python test_basic.py
```

### 2. 训练模型

训练 MLP 和 CNN 模型：
```bash
cd codes
python test_train.py
```

该脚本会：
- 训练 MLP 基线模型（5 个 epoch）
- 训练 CNN 模型（5 个 epoch）
- 将模型保存到 `saved_models/` 目录
- 在 `results/figures/` 中生成学习曲线图

### 3. 测试模型

在测试集上评估已训练的模型：
```bash
cd codes
python test_model.py
```

该脚本会：
- 加载已保存的模型
- 计算测试集准确率
- 生成混淆矩阵
- 将结果保存到 `results/` 目录

### 4. 运行额外实验（Part C）

运行优化与可视化实验：
```bash
cd codes
python additional_experiments.py
```

该脚本会：
- 对比 SGD 与 Momentum 优化器
- 生成混淆矩阵
- 可视化误分类样本
- 可视化卷积核
- 生成各类数字准确率分析

---

## 实现细节

### Part A：MLP 基线

**网络结构：**
```
输入层 (784) -> 全连接层 (600) -> ReLU -> 全连接层 (10) -> 输出
```

**核心实现：**
- `mynn/op.py` 中的 `Linear` 层：前向传播与反向传播
- `mynn/op.py` 中的 `MultiCrossEntropyLoss`：Softmax + 交叉熵损失
- `mynn/models.py` 中的 `Model_MLP`：MLP 模型定义

**超参数设置：**
- 隐藏层单元数：600
- 学习率：0.06
- 批大小：32
- 训练轮数：5
- 权重衰减：1e-4

### Part B：CNN 模型

**网络结构：**
```
输入 (1×28×28)
-> Conv2D(16通道, 3×3卷积核) -> ReLU -> MaxPool(2×2)  # 输出: 16×14×14
-> Conv2D(32通道, 3×3卷积核) -> ReLU -> MaxPool(2×2)  # 输出: 32×7×7
-> Flatten -> 全连接层(1568->128) -> ReLU -> 全连接层(128->10)
```

**核心实现：**
- `mynn/op.py` 中的 `conv2D`：手动实现的二维卷积
- `mynn/op.py` 中的 `MaxPool2D`：最大池化层（含反向传播）
- `mynn/op.py` 中的 `Flatten`：展平层
- `mynn/models.py` 中的 `Model_CNN`：CNN 模型定义

**超参数设置：**
- 学习率：0.05
- 批大小：32
- 训练轮数：5

### Part C：额外探索方向

#### 方向一：优化器（动量 Momentum）

**实现位置：** `mynn/optimizer.py` 中的 `MomentGD`

```python
v_t = mu * v_{t-1} - lr * grad
param = param + v_t
```

**实验内容：**
- 对比 SGD 与 Momentum（mu=0.9）
- 保持相同超参数以确保公平对比
- 衡量收敛速度与最终准确率

#### 方向五：错误分析与可视化

**分析内容包括：**
1. 测试集上的混淆矩阵
2. 误分类样本可视化
3. 卷积核权重可视化
4. 各类数字准确率分析

---

## 实验结果汇总

| 实验 | 模型 | 测试集准确率 |
|------|------|-------------|
| 基线 | MLP | ~97.5% |
| 卷积网络 | CNN | ~98.5% |
| 动量优化 | MLP + Momentum | ~98.0% |

**主要发现：**
1. CNN 以约 10 倍更少的参数超越了 MLP
2. 动量优化器加速了收敛并提升了准确率
3. 最易混淆的数字对为形状相似的组合（如 4/9、3/8）
4. CNN 学习到了层次化的、可解释的特征

---

## 文件说明

### 核心库（`mynn/`）

- **`op.py`**：核心神经网络算子
  - `Linear`：全连接层
  - `conv2D`：二维卷积层
  - `ReLU`：激活函数
  - `MaxPool2D`：最大池化层
  - `Flatten`：展平层
  - `MultiCrossEntropyLoss`：损失函数

- **`models.py`**：模型定义
  - `Model_MLP`：多层感知机
  - `Model_CNN`：卷积神经网络

- **`optimizer.py`**：优化算法
  - `SGD`：随机梯度下降
  - `MomentGD`：带动量的随机梯度下降

- **`lr_scheduler.py`**：学习率调度器
  - `StepLR`：阶梯衰减
  - `MultiStepLR`：多阶梯衰减
  - `ExponentialLR`：指数衰减

- **`runner.py`**：训练与评估循环

- **`metric.py`**：评估指标
  - `accuracy`：分类准确率
  - `confusion_matrix`：混淆矩阵

### 脚本文件

- **`test_train.py`**：主训练脚本
- **`test_model.py`**：模型评估脚本
- **`additional_experiments.py`**：Part C 额外实验
- **`test_basic.py`**：基本功能测试

---

## 注意事项

1. **无外部深度学习框架依赖：** 所有核心算子均使用 NumPy 从零实现，未调用 PyTorch、TensorFlow 等框架。

2. **仅使用 CPU：** 本实现完全在 CPU 上运行，无需 GPU。

3. **数据集：** MNIST 数据集已包含在 `dataset/MNIST/` 目录中。

4. **模型权重：** 训练后的模型保存在 `saved_models/` 目录（不上传至代码仓库）。

5. **实验结果：** 所有图表和结果保存在 `results/figures/` 目录。

---

## 参考文献

1. 课程讲义：神经网络与深度学习
2. MNIST 数据集：http://yann.lecun.com/exdb/mnist/
3. NumPy 文档：https://numpy.org/doc/

---

## 许可

本项目仅用于教学目的。
