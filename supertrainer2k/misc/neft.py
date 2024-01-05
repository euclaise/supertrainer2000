import math
import torch
from torch import nn
from ..utils import patch_model

class NEFTEmbedding(nn.Module):
    def __init__(self, orig_embeddings: nn.Embedding, alpha: float, seq_len: int):
        super().__init__()
        self.emb = orig_embeddings
        self.alpha = alpha
        self.seq_len = seq_len
    def forward(self, xs):
        xs = self.emb(xs)
        eps = torch.rand_like(xs)
        xs = xs + (self.alpha / math.sqrt(self.seq_len * xs.shape[-1])) * eps
        return xs

def apply_neft(model, alpha, seq_len):
    patch_model(model, nn.Embedding, lambda m: NEFTEmbedding(m, alpha, seq_len))


def remove_neft(model):
    patch_model(model, NEFTEmbedding, lambda m: m.emb)
