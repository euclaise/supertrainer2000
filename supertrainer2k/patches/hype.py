import math
import torch
from torch import nn
from ..utils import patch_model
from .quantemb import QuantEmbedding

# https://arxiv.org/abs/2212.08853

class HyPeLayer(nn.Module):
    def __init__(self, orig: nn.Module, sigma: float, normal: bool):
        super().__init__()
        self.orig = orig
        self.sigma = sigma
        self.noise = torch.randn_like if normal else torch.rand_like
    
    def forward(self, hidden_states,**kwargs):
        eps = torch.rand_like(hidden_states)
        xs = hidden_states + self.sigma*eps
        return self.orig(xs, **kwargs)

    def unpatch(self):
        return self.orig
        
def apply_hype(model, layer_class, sigma=1e-5, normal=True):
    assert layer_class is not None
    patch_model(model, [layer_class], lambda m: HyPeLayer(m, sigma, normal))

def remove_hype(model):
    patch_model(model, [HyPeLayer], lambda m: m.unpatch())
