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
    
    def forward(self, xs):
        xs = self.orig(xs)
        eps = torch.rand_like(xs)
        xs = xs + self.sigma*eps
        return xs

    def unpatch(self):
        return self.orig
        
def apply_hype(model, sigma, layer_key='layer'):
    patch_model(model, [nn.Module], lambda m: HyPeLayer(m, sigma), name_constraints = lambda k: k == 'layer_key')

def remove_hype(model):
    patch_model(model, [HyPeLayer], lambda m: m.unpatch())
