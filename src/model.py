import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# T5相对位置编码
class T5RelativePositionalEncoding(nn.Module):
    def __init__(self, max_relative_position=128):
        super().__init__()
        self.max_relative_position = max_relative_position
        # 相对位置偏置嵌入层
        self.relative_attention_bias = nn.Embedding(
            2 * max_relative_position + 1, 1
        )

    def forward(self, q_len, k_len):
        # 创建查询和键的位置索引
        range_vec_q = torch.arange(q_len, device=self.relative_attention_bias.weight.device)
        range_vec_k = torch.arange(k_len, device=self.relative_attention_bias.weight.device)
        # 计算相对位置距离矩阵
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        # 限制相对位置范围
        distance_mat_clamped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        # 将位置索引映射到嵌入索引
        final_mat = distance_mat_clamped + self.max_relative_position
        return self.relative_attention_bias(final_mat).squeeze(-1)

# T5多头注意力机制
class T5MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_relative_position=128):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 相对位置编码
        self.relative_pos_encoding = T5RelativePositionalEncoding(
            max_relative_position
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)

        # 线性变换并重塑为多头形式
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 添加相对位置偏置
        relative_bias = self.relative_pos_encoding(q_len, k_len)
        relative_bias = relative_bias.unsqueeze(0).unsqueeze(0)
        scores += relative_bias.to(scores.device)

        # 应用注意力掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)

        # 合并多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(output)

# T5前馈网络
class T5FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# T5编码器层
class T5EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, max_relative_position=128):
        super().__init__()
        self.self_attention = T5MultiHeadAttention(
            d_model, n_heads, dropout, max_relative_position
        )
        self.ffn = T5FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力加残差连接和层归一化
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络加残差连接和层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

# T5解码器层
class T5DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, max_relative_position=128):
        super().__init__()
        self.self_attention = T5MultiHeadAttention(
            d_model, n_heads, dropout, max_relative_position
        )
        self.cross_attention = T5MultiHeadAttention(
            d_model, n_heads, dropout, max_relative_position
        )
        self.ffn = T5FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力层
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 交叉注意力层
        cross_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))

        # 前馈网络层
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x

# T5编码器
class T5Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1,
                 max_relative_position=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            T5EncoderLayer(d_model, n_heads, d_ff, dropout, max_relative_position)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, mask=None):
        # 词嵌入
        x = self.token_embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.dropout(x)

        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)

        return x

# T5解码器
class T5Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1,
                 max_relative_position=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            T5DecoderLayer(d_model, n_heads, d_ff, dropout, max_relative_position)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 词嵌入
        x = self.token_embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.dropout(x)

        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

# 完整的T5模型
class T5(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048,
                 dropout=0.1, pad_token_id=1, max_relative_position=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # 共享的词嵌入层
        self.shared_embedding = nn.Embedding(vocab_size, d_model)

        # 编码器和解码器
        self.encoder = T5Encoder(
            vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_relative_position
        )
        self.encoder.token_embedding.weight = self.shared_embedding.weight

        self.decoder = T5Decoder(
            vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_relative_position
        )
        self.decoder.token_embedding.weight = self.shared_embedding.weight

        # 输出投影层，与嵌入层共享权重
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.shared_embedding.weight

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    # 权重初始化
    def init_weights(self):
        nn.init.normal_(self.shared_embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # 前向传播
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)

    # 编码方法
    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)

    # 解码方法
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)