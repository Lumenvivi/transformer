#!/bin/bash
# 数据处理（训练Tokenizer+生成JSON格式数据）
python ../src/data_processed.py

# 训练模型（固定随机种子，可直接重现实验）
python ../src/train.py \
  --d_model 256 \
  --n_layers 6 \
  --n_heads 8 \
  --d_ff 1024 \
  --dropout 0.2 \
  --max_relative_position 128 \
  --batch_size 64 \
  --epochs 10 \
  --learning_rate 1e-4 \
  --grad_clip 1.0 \
  --warmup_steps 4000 \
  --max_seq_len 100 \
  --data_dir ../data \
  --save_dir ../checkpoints \
  --log_dir ../results \
  --seed 42 \
  --device cuda
