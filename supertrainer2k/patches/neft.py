import math
import torch
from torch import nn
from ..utils import patch_model
from .quantemb import QuantEmbedding

# https://arxiv.org/abs/2310.05914

class NEFTEmbedding(nn.Module):
    def __init__(self, orig_embeddings: nn.Module, alpha: float, seq_len: int):
        super().__init__()
        self.emb = orig_embeddings
        self.alpha = alpha
        self.seq_len = seq_len
    
    def forward(self, input_ids):
        xs = self.emb(input_ids)
        eps = torch.rand_like(xs)
        xs = xs + (self.alpha / math.sqrt(self.seq_len * xs.shape[-1])) * eps
        return xs

    def unpatch(self):
        return self.emb
        
def apply_neft(model, alpha, seq_len):
    patch_model(model, [nn.Embedding, QuantEmbedding], lambda m: NEFTEmbedding(m, alpha, seq_len))

def remove_neft(model):
    patch_model(model, [NEFTEmbedding], lambda m: m.unpatch())
