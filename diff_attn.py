# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 19:08:48 2025

@author: Jose Antonio
"""

import math

import torch.nn as nn
import torch
from torch.nn.modules.normalization import RMSNorm

""" Implemented from https://arxiv.org/pdf/2410.05258 """

# B -> batch
# N -> number of patches
# d_model -> hidden dim of model (embeddings dimensionality)


class DiffAttn(nn.Module):
    def __init__(self,
                 d_model: int,
                 attention_head_size: int,
                 depth: int, # lambda_init is initialized with a value that depends on l (the index of the layer) [1, L]
                 bias: bool = False  # They dont use it
                 ) -> None:
        super().__init__()
        self.k_proj = nn.Linear(d_model, attention_head_size, bias = bias)
        self.q_proj = nn.Linear(d_model, attention_head_size, bias = bias)
        self.v_proj = nn.Linear(d_model, attention_head_size, bias = bias)
        
        self.attention_head_size = attention_head_size
        
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.attention_head_size//2, dtype=torch.float32))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.attention_head_size//2, dtype=torch.float32))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.attention_head_size//2, dtype=torch.float32))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.attention_head_size//2, dtype=torch.float32))
        
        nn.init.normal_(self.lambda_q1, std=0.1)
        nn.init.normal_(self.lambda_q2, std=0.1)
        nn.init.normal_(self.lambda_k1, std=0.1)
        nn.init.normal_(self.lambda_k2, std=0.1)
        
        self.norm = RMSNorm(self.attention_head_size)
        
        
    def forward(self, x: torch.Tensor):
        
        # x -> (B, N, d_model)
        
        q = self.q_proj(x)  # (B, N, attention_head_size)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Splitting into two matrixes
        q1, q2 = torch.split(q, int(q.shape[2] / 2), dim=2)  # (B, N, d) where d = attention_head_size / 2
        k1, k2 = torch.split(k, int(k.shape[2] / 2), dim=2)
       
        num1 = torch.matmul(q1, k1.transpose(-1, -2))
        den1 = math.sqrt(self.attention_head_size // 2)
        attn1_score = num1 / den1
        attn1_prob = nn.functional.softmax(attn1_score, dim=-1)
        
        num2 = torch.matmul(q2, k2.transpose(-1, -2))
        den2 = math.sqrt(self.attention_head_size // 2)
        attn2_score = num2 / den2
        attn2_prob = nn.functional.softmax(attn2_score, dim=-1)
        
        lambda_final = torch.exp(self.lambda_q1 @ self.lambda_k1) \
            - torch.exp(self.lambda_q2 @ self.lambda_k2) + self.lambda_init
        
        attn_out = (attn1_prob - lambda_final * attn2_prob) @ v
        
        # check equation 3 (its simpler to do this per head instead of doing it in MultiHeadDiffAttn)
        return (1 - self.lambda_init) * self.norm(attn_out)


class MultiHeadDiffAttn(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 depth: int,  # the layer idx of the transformer (1-indexed)
                 dropout_rate: float,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        
        self.attention_head_size = d_model // n_heads
        
        self.heads = nn.ModuleList([DiffAttn(d_model, self.attention_head_size, depth, bias)
                                    for _ in range(n_heads)])
        
        self.proj_d = nn.Linear(  # In case d_model / n_heads is not integer
            n_heads * self.attention_head_size,
            d_model
            )
        
        # dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor):
        
        attn_outs = [head(x) for head in self.heads]
        attn = torch.cat([attn_out for attn_out in attn_outs], dim=-1)  # B, N_patches, n_heads * attn_head_size
        
        attn = self.proj_d(attn)
        attn = self.dropout(attn)
        
        return attn


if __name__ == "__main__":
    
    x = torch.rand((10, 256, 768))
    
    attn = DiffAttn(768, 64, 1)
    
    result = attn(x)
    
    mhda = MultiHeadDiffAttn(768, 12, 2, 0.1)
    
    result = mhda(x)
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        