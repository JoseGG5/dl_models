# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 11:05:17 2025

@author: Jose Antonio
"""

import math

import torch.nn as nn
import torch

from diff_attn import *
from vit import *

""" Changes in ViT arquitecture to implement the DiffAttn mechanism """


class EncoderBlock(nn.Module):
    """ The encoder blocks compose by two layer normalizations, an MLP and MHDiffA (adapted)"""
    def __init__(
            self,
            d: int,
            intermediate_dim: int,
            dropout_rate: float,
            depth: int,  # The index of the transformer layer
            attention_head_size: int = 64,
            n_heads: int = 12
            ) -> None:
        
        super().__init__()
        
        self.norm1 = nn.LayerNorm(normalized_shape=d)
        self.norm2 = nn.LayerNorm(normalized_shape=d)
        self.mha = MultiHeadDiffAttn(
            d_model=d, 
            n_heads=n_heads,
            depth=depth,
            dropout_rate=dropout_rate
            )

        self.mlp = MLP(d, intermediate_dim, dropout_rate) 
    
        
    def forward(self, x: torch.Tensor):
        
        y = self.norm1(x)
        attn_out = self.mha(y)
        y = x + attn_out
        
        z = self.norm2(y)
        z = self.mlp(z)
        out = z + y
        
        return out


""" In the DiffAttn paper they remove the bias.
    In addition, as attention probs are not easy to calculate in Diff Attn,
    this one only returns the attn coefficients"""

class DiffVisionTransformer(nn.Module):
    """ ViT implementation """
    def __init__(
            self,
            d: int,
            img_size: tuple[int],
            patch_size: tuple[int],
            intermediate_dim: int,
            dropout_rate_blocks: float,
            in_channels: int = 1,
            attention_head_size: int = 64,
            n_heads: int = 12,
            n_blocks: int = 12,
            num_classes: int = 10
            ) -> None:
        
        super().__init__()
        
        self.patchifier = PatchEmbed(img_size, patch_size, d, in_channels)
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                self.patchifier.n_patches + 1,  # +1 because of the cls_token
                d)
            )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(
                d, intermediate_dim, dropout_rate_blocks,
                l, attention_head_size, n_heads
                ) for l in range(n_blocks)]
            )
        
        # For classification we use a MLP head that goes from d to num_classes
        self.mlp_head = MLP(d, num_classes, dropout_rate_blocks, classification_head=True)
        
        
    def forward(self, x: torch.Tensor):
        x = self.patchifier(x)  # B, N_patches, D
        
        # cls_token appended at the beginning
        B, _, _ = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)  # basically we repeat over batch dimension
        x = torch.cat((cls_token, x), dim=1)  
        
        # add positional embeddings
        x = self.pos_embed + x
        
        # apply encoder
        all_attns = []
        for block in self.encoder_blocks:
            x = block(x)
        
        x_cls = x[:, 0, :]
        out = self.mlp_head(x_cls)
        
        return (x_cls, out)   # cls token as a feature extractor



if __name__ == "__main__":
    
    x = torch.rand((10, 1, 256, 512))
    
    
    vit = DiffVisionTransformer(
        768,
        (256, 512),
        (16,16),
        4*768,
        0.2,
        num_classes=2)
    
    with torch.no_grad():
        cls_token, logits = vit(x)
    