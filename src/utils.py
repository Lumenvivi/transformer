import torch
import torch.nn as nn
from tokenizers import Tokenizer
import json
import os
import numpy as np
from sacrebleu import corpus_bleu

# 加载tokenizer
def load_tokenizer(tokenizer_path):
    return Tokenizer.from_file(tokenizer_path)


# 创建源掩码和目标掩码
def create_masks(src, tgt, pad_token_id):
    src_mask = (src != pad_token_id).unsqueeze(1).unsqueeze(2)

    tgt_pad_mask = (tgt != pad_token_id).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)

    return src_mask, tgt_mask


# 加载处理后的数据
def load_data(data_dir, split):
    with open(os.path.join(data_dir, f"{split}.json"), 'r', encoding='utf-8') as f:
        return json.load(f)


# 计算BLEU分数
def calculate_bleu(references, hypotheses):
    return corpus_bleu(hypotheses, [references],force=True).score

