import torch
from torch.optim import Optimizer
import warnings
import math

class Adalite(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        eps: float = 1e-5,
        eps2: float = 1e-3,
        min_trust_ratio: float = 1e-3,
        Lambda: float = 0.01,
        beta_decay: float = 0.8,
        centralize: bool = True,
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            eps2=eps2,
            min_trust_ratio=min_trust_ratio,
            Lambda=Lambda,
            weight_decay=weight_decay,
            beta_decay=beta_decay,
            centralize=centralize
        )

        super(Adalite, self).__init__(params, defaults)


    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                alpha = group['lr']

                if p.grad is None:
                    continue
                
                g = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0

                    state['c'] = torch.zeros_like(p.mean(dim=tuple(range(len(p.shape) - 1)), keepdim=False))

                state['step'] += 1

                if group['centralize'] and sum(g.shape) > 1:
                    g.sub_(g.mean(dim=tuple(range(1, len(g.shape))), keepdim=True))

                beta_t = 1.0 - math.pow(state['step'], group['beta_decay'])
                v = g.square()

                c_e = state['c']
                while c_e.dim() < g.dim():
                    c_e = c_e.unsqueeze(0)

                v.mul_(beta_t).add_(c_e.broadcast_to(g.shape), alpha=1-beta_t)
                state['c'] = v.mean(dim=tuple(range(len(v.shape) - 1)), keepdim=False) # Take mean over all dims except first
                v.add_(group['eps'])

                m = v.rsqrt() * g
                    
                p_norm = p.norm()
                g_norm = g.norm()

                if p_norm != 0. and g_norm != 0.:
                    trust_ratio = (p_norm / g_norm.clamp(min=group['eps2'])).clamp(min=group['min_trust_ratio'])
                    u.mul_(trust_ratio)
                else:
                    trust_ratio = 1.0

                u.add_(p.data, alpha=group['Lambda'])

                u.add_(p.data, alpha=group['Lambda'])

                p.data.add_(m, alpha=-alpha)
        return loss
