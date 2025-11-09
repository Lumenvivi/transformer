import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import time
import math
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from model import T5
from utils import load_tokenizer, load_data, create_masks, calculate_bleu

# Noam学习率调度器
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, step_num=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = step_num

    def step(self):
        self.step_num += 1
        # 计算当前学习率
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_state_dict(self):
        return {
            'step_num': self.step_num,
            'warmup_steps': self.warmup_steps,
            'd_model': self.d_model
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']
        self.warmup_steps = state_dict['warmup_steps']
        self.d_model = state_dict['d_model']

# 翻译数据集类
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=100, is_training=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        en_text = item['en']
        de_text = item['de']
        # 对文本进行编码
        en_encoding = self.tokenizer.encode(en_text)
        de_encoding = self.tokenizer.encode(de_text)
        # 添加特殊标记并截断
        en_tokens = [2] + en_encoding.ids[:self.max_seq_len - 2] + [3]
        de_tokens = [2] + de_encoding.ids[:self.max_seq_len - 2] + [3]
        # 填充到最大长度
        en_tokens = en_tokens + [1] * (self.max_seq_len - len(en_tokens))
        de_tokens = de_tokens + [1] * (self.max_seq_len - len(de_tokens))
        return {
            'src': torch.tensor(en_tokens, dtype=torch.long),
            'tgt': torch.tensor(de_tokens, dtype=torch.long)
        }

# 贪婪解码函数
def greedy_decode(model, src, src_mask, tokenizer, max_len, device, pad_token_id=1):
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        # 初始化目标序列，以开始标记开始
        ys = torch.ones(src.size(0), 1).fill_(2).long().to(device)
        for i in range(max_len - 1):
            # 创建因果掩码
            tgt_mask = torch.tril(torch.ones((ys.size(1), ys.size(1)), device=device)).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
            tgt_mask = tgt_mask.expand(ys.size(0), -1, -1, -1)
            out = model.decode(ys, encoder_output, src_mask, tgt_mask)
            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            # 如果所有序列都生成了结束标记，则停止
            if (next_word == 3).all():
                break
    # 将token序列转换回文本
    translations = []
    for seq in ys:
        tokens = seq.tolist()
        if 3 in tokens:
            tokens = tokens[1:tokens.index(3)]
        else:
            tokens = tokens[1:]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        translations.append(text)
    return translations

# 训练一个epoch
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, pad_token_id, grad_clip=1.0):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in tqdm(dataloader, desc="Training"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask, tgt_mask = create_masks(src, tgt[:, :-1], pad_token_id)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)
        # 计算损失，忽略填充标记
        loss = criterion(output.contiguous().view(-1, output.size(-1)),
                         tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches

# 评估函数
def evaluate(model, dataloader, criterion, device, pad_token_id, tokenizer, max_seq_len, num_examples=1):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_references = []
    all_hypotheses = []
    all_sources = []
    example_sources = []
    example_references = []
    example_hypotheses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_mask, tgt_mask = create_masks(src, tgt[:, :-1], pad_token_id)
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)
            loss = criterion(output.contiguous().view(-1, output.size(-1)),
                             tgt[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
            n_batches += 1
            # 生成翻译结果
            hypotheses = greedy_decode(model, src, src_mask, tokenizer, max_seq_len, device, pad_token_id)
            references = [tokenizer.decode(tgt[i].tolist(), skip_special_tokens=True) for i in range(tgt.size(0))]
            sources = [tokenizer.decode(src[i].tolist(), skip_special_tokens=True) for i in range(src.size(0))]
            all_hypotheses.extend(hypotheses)
            all_references.extend(references)
            all_sources.extend(sources)
            # 收集样例
            if len(example_sources) < num_examples:
                remaining = num_examples - len(example_sources)
                example_sources.extend(sources[:remaining])
                example_references.extend(references[:remaining])
                example_hypotheses.extend(hypotheses[:remaining])

    # 计算BLEU分数
    bleu_score = calculate_bleu(all_references, all_hypotheses)
    examples = list(zip(example_sources, example_references, example_hypotheses))
    return total_loss / n_batches, bleu_score, examples

# 获取训练参数解析器
def get_training_parser():
    parser = argparse.ArgumentParser(description="Train T5 with relative positional encoding")
    # 模型参数
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of encoder/decoder layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="FFN hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--max_relative_position", type=int, default=128, help="Max relative position for T5")
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate (for optimizer)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Warmup steps for Noam scheduler")
    # 数据参数
    parser.add_argument("--data_dir", default="../data", help="Data directory")
    parser.add_argument("--max_seq_len", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--save_dir", default="../checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", default="../results", help="Directory for logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_checkpoint", default="best_model.pth", help="Checkpoint file to resume from")
    return parser

# 加载检查点
def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    # 加载优化器状态
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 加载调度器状态
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    train_losses = checkpoint.get('train_losses', [])
    best_bleu = checkpoint.get('best_bleu', [])
    val_losses = checkpoint.get('val_losses', [])
    val_bleus = checkpoint.get('val_bleus', [])
    learning_rates = checkpoint.get('learning_rates', [])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
    return start_epoch, best_bleu, train_losses, val_losses, val_bleus, learning_rates

# 保存检查点
def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, val_bleus, learning_rates,
                    best_bleu, args, is_best=False, filename=None):
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"

    checkpoint_path = os.path.join(args.save_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.get_state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_bleus': val_bleus,
        'learning_rates': learning_rates,
        'best_bleu': best_bleu,
        'args': vars(args)
    }

    torch.save(checkpoint, checkpoint_path)

    # 如果是最好模型，额外保存一份
    if is_best:
        best_path = os.path.join(args.save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved with BLEU: {best_bleu:.2f}")

    print(f"Checkpoint saved: {checkpoint_path}")

# 主训练函数
def main():
    parser = get_training_parser()
    args = parser.parse_args()
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # 加载tokenizer和数据
    tokenizer = load_tokenizer(os.path.join(args.data_dir, "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer loaded, vocab size: {vocab_size}")
    train_data = load_data(args.data_dir, 'train')
    val_data = load_data(args.data_dir, 'validation')
    test_data = load_data(args.data_dir, 'test')
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    # 创建数据集和数据加载器
    train_dataset = TranslationDataset(train_data, tokenizer, args.max_seq_len, True)
    val_dataset = TranslationDataset(val_data, tokenizer, args.max_seq_len, False)
    test_dataset = TranslationDataset(test_data, tokenizer, args.max_seq_len, False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # 初始化模型
    model = T5(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_token_id=1,
        max_relative_position=args.max_relative_position
    ).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    scheduler = NoamScheduler(optimizer, args.d_model, args.warmup_steps)
    # 初始化训练变量
    start_epoch = 0
    best_bleu = 0
    train_losses = []
    val_losses = []
    val_bleus = []
    learning_rates = []
    # 恢复训练
    if args.resume:
        checkpoint_path = os.path.join(args.save_dir, args.resume_checkpoint)
        try:
            start_epoch, best_bleu, train_losses, val_losses, val_bleus, learning_rates = load_checkpoint(
                model, optimizer, scheduler, checkpoint_path, device
            )
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            args.resume = False
    # 模型预热（非恢复训练时）
    if not args.resume:
        test_batch = next(iter(train_loader))
        src = test_batch['src'][:2].to(device)
        src_mask = (src != 1).unsqueeze(1).unsqueeze(2)
    # 开始训练循环
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device,
                                 pad_token_id=1, grad_clip=args.grad_clip)
        train_losses.append(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        # 在验证集上评估
        val_loss, val_bleu, _ = evaluate(model, val_loader, criterion, device,
                                                    pad_token_id=1, tokenizer=tokenizer,
                                                    max_seq_len=args.max_seq_len, num_examples=0)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val BLEU: {val_bleu:.2f}, Current LR: {current_lr:.6f}")
        # 保存最佳模型
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses,
                            val_bleus, learning_rates, best_bleu, args, is_best=True)

    # 保存最终模型
    save_checkpoint(args.epochs - 1, model, optimizer, scheduler, train_losses, val_losses,
                    val_bleus, learning_rates, best_bleu, args,
                    filename="final_model.pth")
    # 在测试集上评估最佳模型
    print("\nTesting best model...")
    checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_bleu, test_examples = evaluate(model, test_loader, criterion, device,
                                                       pad_token_id=1, tokenizer=tokenizer,
                                                       max_seq_len=args.max_seq_len, num_examples=3)
        print(f"Test Loss: {test_loss:.4f}, Test BLEU: {test_bleu:.2f}")
        print("\n测试集样例:")
        for i, (source, reference, hypothesis) in enumerate(test_examples, 1):
            print(f"样例 {i}:")
            print(f"源文本: {source}")
            print(f"参考翻译: {reference}")
            print(f"模型翻译: {hypothesis}\n")
    else:
        print("Best model checkpoint not found, using current model for testing")
        test_loss, test_bleu, test_examples = evaluate(model, test_loader, criterion, device,
                                                       pad_token_id=1, tokenizer=tokenizer,
                                                       max_seq_len=args.max_seq_len, num_examples=3)
        print(f"Test Loss: {test_loss:.4f}, Test BLEU: {test_bleu:.2f}")
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_bleus, learning_rates, args.log_dir)
    # 保存结果
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_bleus': val_bleus,
        'test_bleu': test_bleu,
        'test_loss': test_loss,
        'best_val_bleu': best_bleu,
        'learning_rates': learning_rates
    }
    with open(os.path.join(args.log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    result_data = {
        'final_train_loss': [round(results['train_losses'][-1], 4)],
        'final_val_loss': [round(results['val_losses'][-1], 4)],
        'best_val_bleu': [round(results['best_val_bleu'], 4)],
        'test_bleu': [round(results['test_bleu'], 4)]
    }

    result_df = pd.DataFrame(result_data)

    # 保存到CSV文件
    results_file = os.path.join(args.log_dir, "final_results.csv")
    # 如果文件已存在，则追加数据，否则创建新文件
    if os.path.exists(results_file):
        result_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(results_file, index=False)

    print(f"\nTraining completed! Best validation BLEU: {best_bleu:.2f}")
    print(f"Test BLEU: {test_bleu:.2f}")
    print(f"Final results saved to: {results_file}")

# 绘制训练曲线
def plot_training_curves(train_losses, val_losses, val_bleus, learning_rates, log_dir):
    # 绘制损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(fontsize=16)
    plt.title('Training and Validation Loss', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # 绘制BLEU分数曲线
    plt.figure(figsize=(8, 6))
    plt.plot(val_bleus, 'g-', label='Val BLEU', linewidth=2, marker='o', markersize=4)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('BLEU Score', fontsize=18)
    plt.title('Validation BLEU Score', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'bleu_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Training curves saved as images.")

if __name__ == "__main__":
    main()