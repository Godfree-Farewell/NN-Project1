# MNIST 手写数字分类模型

## 模型概述

本项目是复旦大学《神经网络与深度学习》课程 Project 1 的实现。仅使用 NumPy 从零搭建了自定义神经网络库，在 MNIST 手写数字分类任务上训练了多层感知机（MLP）和卷积神经网络（CNN）模型。

- **作者：** 龙政宇
- **学号：** 22307130289
- **框架：** 纯 NumPy 实现（无 PyTorch/TensorFlow）
- **任务类型：** 图像分类
- **数据集：** MNIST（60,000 训练图像，10,000 测试图像）

## 模型介绍

### 模型架构

#### 1. MLP（多层感知机）
```
输入 (784) -> 全连接 (600) -> ReLU -> 全连接 (10) -> 输出
```
- 参数量：约 471K
- 验证集准确率：90.80%
- 测试集准确率：91.59%

#### 2. CNN（卷积神经网络）
```
输入 (1x28x28)
-> Conv2D(1->16, 3x3, pad=1) -> ReLU -> MaxPool(2x2)  # 16x14x14
-> Conv2D(16->32, 3x3, pad=1) -> ReLU -> MaxPool(2x2)  # 32x7x7
-> Flatten -> 全连接(1568->128) -> ReLU -> 全连接(128->10)
```
- 参数量：约 109K
- 验证集准确率：98.94%
- 测试集准确率：98.91%

### 核心实现

- **Linear 层：** 手动实现前向传播和反向传播
- **Conv2D：** 使用 im2col + 矩阵乘法向量化实现
- **MultiCrossEntropyLoss：** 合并 Softmax 和交叉熵，保证数值稳定性
- **优化器：** SGD、Momentum (mu=0.9)
- **学习率调度：** MultiStepLR

## 模型训练

### 环境要求
```bash
Python >= 3.8
NumPy
Matplotlib（可视化）
```

### 训练命令

#### 训练 MLP 基线
```bash
cd codes
python test_train.py --model mlp --lr 0.01 --epochs 5
```

#### 训练 CNN
```bash
cd codes
python test_train.py --model cnn --lr 0.05 --epochs 10
```

### 超参数设置

| 参数 | MLP | CNN |
|------|-----|-----|
| 学习率 | 0.01 | 0.05 |
| 批大小 | 32 | 32 |
| 训练轮数 | 5 | 10 |
| 权重初始化 | He 初始化 | He 初始化 |
| 学习率调度 | MultiStepLR [800,2400,4000] | MultiStepLR [2000,5000,7000] |

## 模型推理

### 加载模型

```python
import sys
sys.path.append('codes')
from mynn.models import Model_CNN, Model_MLP

# 加载 CNN 模型
model = Model_CNN()
model.load_model('cnn/best_model.pickle')

# 或加载 MLP 模型
model = Model_MLP()
model.load_model('mlp/best_model.pickle')
```

### 单张图片推理

```python
import numpy as np

# 准备输入（28x28 灰度图像，归一化到 [0,1]）
image = np.random.rand(1, 1, 28, 28)  # [batch, channel, H, W]

# 前向传播
output = model(image)
prediction = np.argmax(output, axis=1)
print(f"预测结果: {prediction[0]}")
```

### 测试集评估

```bash
cd codes
python test_model.py --model_path saved_models/cnn/best_model.pickle
```

## 模型效果

### 主要结果对比

| 模型 | 优化器 | 参数量 | 验证集准确率 | 测试集准确率 |
|------|--------|--------|------------|------------|
| MLP | SGD | ~471K | 90.80% | 91.59% |
| MLP | Momentum | ~471K | 96.07% | - |
| CNN | SGD | ~109K | 98.94% | 98.91% |

### 关键发现

1. **CNN 显著优于 MLP**：在参数量减少 4 倍的情况下，准确率提升 7.32%
2. **Momentum 加速收敛**：相比 SGD，收敛速度提升约 2 倍，最终准确率提升 5.27%
3. **权重初始化至关重要**：使用 He 初始化使深层 CNN 能够正常训练

### 可视化结果

- 学习曲线（MLP & CNN）
- 混淆矩阵
- 卷积核可视化
- 错分样本分析

## 文件说明

| 文件 | 说明 |
|------|------|
| `cnn/best_model.pickle` | CNN 最佳模型权重 |
| `mlp/best_model.pickle` | MLP 最佳模型权重 |
| `momentum/best_model.pickle` | MLP + Momentum 模型权重 |
| `sgd/best_model.pickle` | MLP + SGD 模型权重 |

## 相关链接

- **GitHub 代码仓库：** https://github.com/Godfree-Farewell/NN-Project1
- **课程：** 复旦大学《神经网络与深度学习》

## 许可证

Apache 2.0
