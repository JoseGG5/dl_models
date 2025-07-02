# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 20:30:44 2025

@author: Jose Antonio
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
import numpy as np

from vit import VisionTransformer

""" Only for ViT as the best performance comes with ViT and all of the
    CLIP variants over the years have been using ViT. As the text encoder
    I just selected DistilBERT. """

def get_image_encoder(
        d: int,
        img_size: tuple[int],
        patch_size: tuple[int],
        intermediate_dim: int,  # Typically 4*d
        dropout_rate_blocks: float
        ) -> VisionTransformer:
    
    return VisionTransformer(
        d, img_size, patch_size,
        intermediate_dim, dropout_rate_blocks
        )


def get_text_encoder() -> DistilBertModel:
    cfg = DistilBertConfig()
    return DistilBertModel(cfg)


def clip_loss(logits: torch.tensor, batch_size: int):
    """
    Clip loss that combines images losses and text losses.

    Parameters
    ----------
    logits : torch.tensor
        A [batch_size, batch_size] shape tensor where each row is an image
        and each column is a sentence. Image i corresponds to text i.
    batch_size : int
        The number of elements (pairs image text) in each batch.

    Returns
    -------
    Loss item.

    """
    labels = torch.arange(batch_size)  # The true labels (beacuse for row i), correct label is labels[i] (which is i)
    
    # We basically compute the loss for each image (row). Labels here is correct because the true label for row i is column i.
    loss_i = F.cross_entropy(logits, labels)  # axis = 0 
    loss_t = F.cross_entropy(logits.T, labels)  # equivalent to axis = 1

    loss = (loss_i + loss_t) / 2.0
    
    return loss
     

class Clip(nn.Module):
    def __init__(
            self,
            d_img: int,  # dimensionality of the embeddings from the img_enc
            d_txt: int,  # dimensionality of the embeddings from the txt_enc
            proj_d: int,  # dimensionality of the shared embeddings space
            image_encoder: VisionTransformer,  # img encoder
            text_encoder: DistilBertModel,  # txt encoder
            ) -> None:
        super().__init__()
        
        self.image_encoder = image_encoder
        self.projection_vision = nn.Linear(d_img, proj_d, bias=False)
        self.text_encoder = text_encoder
        self.projection_text = nn.Linear(d_txt, proj_d, bias=False)
        
        # As temp should always be bigger than 0 we store it as a log and later on the forward we apply exp (F^-1)
        self.temperature = nn.Parameter(torch.tensor(np.log(1/0.07)))  
    
    
    def forward(self, x: dict):
        img = x["image"]
        input_ids, mask = x["input_ids"], x["mask"]
        
        cls_token, logits, _ = self.image_encoder(img)  # cls_token -> [B, d]
        x_img = self.projection_vision(cls_token)  
        x_img = F.normalize(x_img, p=2.0, dim=1)  # L2 norm
        
        emb_txt = self.text_encoder(input_ids, mask)["last_hidden_state"]  # [B, T, d]
        emb_txt = emb_txt[:, 0, :]
        x_txt = self.projection_text(emb_txt)
        x_txt = F.normalize(x_txt, p=2.0, dim=1)  # L2 norm
        
        # scaled pairwise cosine similarity
        # (basically rows are images and columns are texts and we get a measure of how close image i is to text j)
        logits = x_img @ x_txt.T * self.temperature.exp()  # [B, B]
        
        return logits
        
    
    
if __name__ == "__main__":
    x = torch.rand((1, 1, 256, 512))
    
    vit = VisionTransformer(
        768,
        (256, 512),
        (16,16),
        4*768,
        0.2,
        output_attentions_probs=False,
        num_classes=2)
    vit.eval()
    
    text = ["My name is Jose"]
    
    txt_encoder = get_text_encoder()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    text_token = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    txt_enc(input_ids = text_token["input_ids"], attention_mask=text_token["attention_mask"])["last_hidden_state"].shape
    
    clip_model = Clip(768, 768, 768, vit, txt_encoder)    
    clip_model.eval()
    
    data = {"image": x, "input_ids": text_token["input_ids"], "mask": text_token["attention_mask"]}
    
    with torch.no_grad():
        clip_model(data)
        