# 基于相对位置偏置的 Transformer Encoder–Decoder 架构实现

## 项目简介

本项目基于 PyTorch 从零实现了一个完整的 Transformer 编码器 - 解码器架构，用于英德机器翻译任务。核心目标是深入理解 Transformer 核心组件原理，掌握序列到序列建模流程，并通过实验验证模型性能。

项目严格遵循课程作业要求，实现了多头自注意力、位置感知前馈网络、残差连接 + 层归一化、相对位置编码等关键模块，同时包含完整的训练 pipeline、消融实验支持和结果可视化功能。

## 核心特性

- 完整实现 Transformer 编码器 - 解码器架构，支持序列到序列任务
- 采用 T5 风格相对位置编码，提升长序列建模能力
- 集成 Noam 学习率调度、梯度裁剪、AdamW 优化器等训练稳定性技巧
- 支持模型 checkpoint 保存 / 加载、训练曲线可视化、BLEU 分数评估
- 提供一键运行脚本，实验结果可完全复现
- 代码结构清晰，关键模块附带详细注释

## 环境配置

### 硬件要求

- 推荐 GPU（CUDA 支持）：训练速度提升 10-20 倍
- 显存要求：≥4GB（默认 batch size=64 时）
- CPU 训练兼容（但训练时间较长，约为 GPU 的 15 倍）

### 依赖安装

```bash
# 创建虚拟环境（可选）
conda create -n transformer python=3.10
conda activate transformer

# 安装依赖包
pip install -r requirements.txt
```

## 代码结构

```plaintext
├── src/                  # 核心代码目录
│   ├── data_processed.py # 数据处理与Tokenizer训练
│   ├── model.py          # Transformer模型实现
│   ├── train.py          # 训练与评估主程序
│   └── utils.py          # 工具函数（掩码、BLEU计算等）
├── scripts/              # 运行脚本目录
│   └── run.sh            # 一键运行脚本（数据处理+训练）
├── checkpoints/          # 模型 checkpoint 目录（自动生成）
├── results/              # 实验结果目录（自动生成）
│   ├── loss_curves.png   # 训练/验证损失曲线
│   ├── bleu_curves.png   # 验证集 BLEU 曲线
│   └── results.json      # 量化结果记录
│   └── final_results.csv # 结果记录表格
├── requirements.txt      # 依赖清单
└── README.md             # 项目说明文档
```

## 运行方法

1. 数据处理（训练 Tokenizer + 生成 JSON 格式数据）

```bash
python src/data_processed.py --data_dir ../data
```

2. 模型训练

```bash
python src/train.py \
```

3. 恢复训练（从最优模型继续）

```bash
python ../src/train.py \
  --resume \
  --resume_checkpoint ../checkpoints/best_model.pth
```

## 数据集说明

本项目使用 **IWSLT2017 (EN ↔ DE)** 数据集，适用于小规模机器翻译任务：

- 任务类型：英德双向机器翻译
- 数据规模：约 20 万训练样本对
- 数据来源：Hugging Face Datasets（自动下载）

## 模型架构

### 核心组件

1. **T5 多头注意力机制**：支持自注意力（编码器 / 解码器）和交叉注意力（解码器 - 编码器）
2. **相对位置编码**：替代传统正弦位置编码，更好捕捉序列相对位置关系
3. **位置感知前馈网络**：GELU 激活函数，提升模型非线性表达能力
4. **残差连接 + 层归一化**：稳定深层模型训练，加速收敛
5. **共享词嵌入**：编码器、解码器、输出投影层共享词嵌入权重，减少参数总量

### 关键超参数

| 参数                  | 默认值 | 说明                 |
| --------------------- | ------ | -------------------- |
| d_model               | 256    | 模型隐藏层维度       |
| n_layers              | 6      | 编码器 / 解码器层数  |
| n_heads               | 8      | 注意力头数           |
| d_ff                  | 1024   | 前馈网络隐藏层维度   |
| dropout               | 0.2    | dropout 概率         |
| max_relative_position | 128    | 最大相对位置编码范围 |
| max_seq_len           | 100    | 序列最大长度         |
| batch_size            | 64     | 训练批次大小         |
| epochs                | 10     | 训练轮数             |
| lr                    | 0.001  | 学习率               |
| warmup_steps          | 4000   | Noam 调度器预热步数  |

## 实验结果

### 量化指标

| final_train_loss | final_val_loss | best_val_bleu | test_bleu |
| ---------------- | -------------- | ------------- | --------- |
| 2.6542           | 2.3671         | 22.72         | 23.16     |

### 可视化结果

训练完成后自动生成以下图表（保存至 `results/` 目录）：

- `loss_curves.png`：训练 / 验证损失随 epoch 变化曲线

- `bleu_curves.png`：验证集 BLEU 分数随 epoch 变化曲线


