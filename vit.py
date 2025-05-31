# -*- coding: utf-8 -*-
"""
Created on Sat May 31 09:43:02 2025

@author: Jose Antonio
"""

import math

import torch
import torch.nn as nn

""" Own implementation of Vision Transformer from: https://arxiv.org/pdf/2010.11929
    The script contains all the necessary modules to build a ViT
    (even Attention and MultiHeadAttention)
    """

class PatchEmbed(nn.Module):
    """ Patchify and flatten an image.
        Takes a tensor with shape B, C, H, W
        Return a tensor with shape B, N_patchs, D
        """
    def __init__(
            self,
            img_size: tuple[int],
            patch_size: tuple[int],
            d: int,
            in_channels: int = 1  # 1 for specs, 3 for images
            ) -> None:
        super().__init__()
        
        patches_height = img_size[0] / patch_size[0]
        patches_width = img_size[1] / patch_size[1]
        self.n_patches = int(patches_height * patches_width)
        
        self.patchify = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d,
            kernel_size=patch_size,
            stride=patch_size
            )
               
    def forward(self, x: torch.Tensor):
        
        # B -> batch size
        # D -> embedding dimensionality
        # P_H -> patches in height dimension
        # P_W -> patches in weight dimension
        # N_patches -> number of patches
        
        x = self.patchify(x)  # B, D, P_H, P_W
        x = torch.flatten(x, start_dim=2, end_dim=-1)  # B, D, N_patches
        x = x.transpose(1, 2)  # B, N_patches, D
        
        return x


class Attention(nn.Module):
    """ Scaled dot product attention.
    Follows formula: attn = softmax(query*key_t / sqrt(d_key)) * value"""
    def __init__(
            self,
            d: int,
            attention_head_size: int,  # The size that a head attends to (head_size = d/num_heads for multihead)
            bias: bool = True
            ) -> None:
        
        super().__init__()
        
        self.q_proj = nn.Linear(d, attention_head_size, bias=bias)
        self.k_proj = nn.Linear(d, attention_head_size, bias=bias)
        self.v_proj = nn.Linear(d, attention_head_size, bias=bias)
        
        self.attention_head_size = attention_head_size
        
    def forward(self, x: torch.Tensor):
        
        # attn = softmax(query*key_t / sqrt(d_key)) * value
        
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        
        num = torch.matmul(query, key.transpose(-1, -2))
        den = math.sqrt(self.attention_head_size)
        
        attn_score = num / den
        
        # Normalize with softmax. The idea is that each row (query) needs to
        # sum 1 so we apply it to columns
        attn_prob = nn.functional.softmax(attn_score, dim=-1)  # B, N_patches, N_patches
        
        # Multiply by values to get the final attention
        attn_out = torch.matmul(attn_prob, value)  # B, N_patches, attn_head_size
        
        return attn_out, attn_prob
        

class MultiHeadAttention(nn.Module):
    """ Basically n attention modules (heads)
    whose outputs get concatenated in the final dimension (head_size)"""
    def __init__(
            self,
            d: int,
            dropout_rate: float,
            attention_head_size: int = 64,
            n_heads: int = 12,
            bias: bool = True
            ) -> None:
        
        super().__init__()
        
        heads = []
        for _ in range(n_heads):
            heads.append(Attention(d, attention_head_size, bias))
        
        self.heads = nn.ModuleList(heads)
        
        # In case one makes d different from attn_head_size * n_heads
        # Also good for the model (more parameters)
        self.proj_d = nn.Linear(n_heads * attention_head_size, d)
        
        # dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor, output_attentions_probs = False):
        
        attn_outs = [head(x) for head in self.heads]
        attn = torch.cat([attn_out for attn_out, _ in attn_outs], dim=-1)  # B, N_patches, n_heads * attn_head_size
        
        attn = self.proj_d(attn)
        attn = self.dropout(attn)
        
        if not output_attentions_probs:        
            return (attn, None)
        
        else:
            # stack creates a new dimension that represents the heads
            attn_probs = torch.stack([attn_probs for _, attn_probs in attn_outs], dim=1)  # B, n_heads, N_patches, N_patches
            return (attn, attn_probs)


class MLP(nn.Module):
    """ A simple multilayer percentron """
    def __init__(
            self,
            d: int,
            intermediate_dim: int,  # Tipically 4*d
            dropout_rate: float,
            classification_head: bool = False  # If false, it is an mlp to be used in the blocks, if true, its the mlp head for classification
            ) -> None:
        
        super().__init__()
        
        self.proj1 = nn.Linear(d, intermediate_dim)
        self.gelu = nn.GELU()
        self.proj2 = nn.Linear(intermediate_dim, d)
        self.dropout = nn.Dropout(dropout_rate)
        self.classification_head = classification_head
        
    def forward(self, x: torch.Tensor):
        
        x = self.proj1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        if not self.classification_head:
            x = self.proj2(x)
            return self.dropout(x)
        
        return x


class EncoderBlock(nn.Module):
    """ The encoder blocks compose by two layer normalizations, an MLP and MHA """
    def __init__(
            self,
            d: int,
            intermediate_dim: int,
            dropout_rate: float,
            attention_head_size: int = 64,
            n_heads: int = 12,
            bias: bool = True,
            output_attentions_probs: bool = False
            ) -> None:
        
        super().__init__()
        
        self.norm1 = nn.LayerNorm(normalized_shape=d)
        self.norm2 = nn.LayerNorm(normalized_shape=d)
        self.mha = MultiHeadAttention(d, dropout_rate, attention_head_size, n_heads, bias)
        self.mlp = MLP(d, intermediate_dim, dropout_rate) 
        
        self.output_attentions_probs = output_attentions_probs
        
    def forward(self, x: torch.Tensor):
        
        y = self.norm1(x)
        attn_out, attn_prob = self.mha(y, output_attentions_probs=self.output_attentions_probs)
        y = x + attn_out
        
        z = self.norm2(y)
        z = self.mlp(z)
        out = z + y
        
        if self.output_attentions_probs:       
            return (out, attn_prob)
        
        return (out, None)
        

class VisionTransformer(nn.Module):
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
            qkv_bias: bool = True,
            output_attentions_probs: bool = False,
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
                attention_head_size, n_heads, qkv_bias,
                output_attentions_probs
                ) for _ in range(n_blocks)]
            )
        
        # For classification we use a MLP head that goes from d to num_classes
        self.mlp_head = MLP(d, num_classes, dropout_rate_blocks, classification_head=True)
        
        self.output_attentions_probs = output_attentions_probs
        
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
            x, attn_prob = block(x)
            if self.output_attentions_probs:
                all_attns.append(attn_prob)
        
        x_cls = x[:, 0, :]
        out = self.mlp_head(x_cls)
        
        if self.output_attentions_probs:
            return (x_cls, out, all_attns)  # cls token as a feature extractor
        
        return (x_cls, out, None)   # cls token as a feature extractor
        
    
if __name__ == "__main__":
    
    x = torch.rand((10, 1, 256, 512))
    
    vit = VisionTransformer(
        768,
        (256, 512),
        (16,16),
        4*768,
        0.2,
        output_attentions_probs=False,
        num_classes=2)
    
    with torch.no_grad():
        cls_token, logits, attn_probs = vit(x)
    
    
    
    
    
    
    
    

    
    
