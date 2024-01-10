import math
import torch
from torch import nn
from ..utils import patch_model
from .quantemb import QuantEmbedding

# https://arxiv.org/abs/2212.08853

class HyPeLayer(nn.Module):
    def __init__(self, orig: nn.Module, sigma: float):
        super().__init__()
        self.orig = orig
        self.sigma = sigma
    
    def forward(self, hidden_states, **kwargs):
        eps = torch.rand_like(hidden_states)
        xs = hidden_states + self.sigma*eps
        return self.orig(xs)

    def unpatch(self):
        return self.orig
        
def apply_hype(model, sigma, layer_class):
    patch_model(model, [layer_class], lambda m: HyPeLayer(m, sigma))

def remove_hype(model):
    patch_model(model, [HyPeLayer], lambda m: m.unpatch())
