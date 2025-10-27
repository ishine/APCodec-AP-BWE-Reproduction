import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import json
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import glob

class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff_out = self.ffn(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x

class AudioTransformer(nn.Module):
    def __init__(self, vocab_size=3073, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.2, max_len=1155, pad_id=4096):
        super().__init__()
        self.max_len = max_len
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pad_mask = (x == self.pad_id)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=pad_mask)
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits #[B, T, vocab_size]

def count_parameters(model):
  """计算 PyTorch 模型的总参数量"""
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return total_params, trainable_params

# 实例化模型 (使用您 train.py 中的参数)
model_to_count = AudioTransformer(
    vocab_size=3073,
    d_model=256,
    nhead=4,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    max_len=1155,
    pad_id=3072
)

# 计算并打印参数量
total_params, trainable_params = count_parameters(model_to_count)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")