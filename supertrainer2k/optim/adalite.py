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
        weight_decay: float = 0.01,
        beta_decay: float = 0.8,
        centralize: bool = True,
        use_rms: bool = True,
        momentum_beta: float = 0.0
    ):
        if momentum_beta != 0.0:
            warnings.warn("Note that Adalite with a non-zero momentum_beta takes significantly more memory than default Adalite.")

        assert eps >= 0. and eps < 1., "Invalid eps value"
        assert weight_decay >= 0. and weight_decay <= 1., "Invalid weight_decay value"
        assert beta_decay >= 0. and beta_decay <= 1., "Invalid beta_decay value"
        assert momentum_beta >= 0. and momentum_beta <= 1., "Invalid momentum_beta value"

        defaults = dict(
            lr=lr,
            eps=eps,
            eps2=eps2,
            min_trust_ratio=min_trust_ratio,
            weight_decay=weight_decay,
            beta_decay=beta_decay,
            centralize=centralize,
            use_rms=use_rms,
            momentum_beta=momentum_beta
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
                    # Initialize state
                    state['step'] = 0

                    state['c'] = torch.zeros_like(p.mean(dim=tuple(range(len(p.shape) - 1)), keepdim=False))
                    if group['momentum_beta'] != 0.0:
                        state['m'] = torch.zeros_like(p)
                    ###

                state['step'] += 1

                if group['centralize'] and sum(g.shape) > 1:
                    g.sub_(g.mean(dim=tuple(range(1, len(g.shape))), keepdim=True))

                beta_t = 1.0 - math.pow(state['step'], -group['beta_decay'])
                u = g.square()

                c_e = state['c']
                while c_e.dim() < g.dim():
                    c_e = c_e.unsqueeze(0)

                u.mul_(1-beta_t).add_(c_e.broadcast_to(g.shape), alpha=beta_t)
                state['c'] = u.mean(dim=tuple(range(len(u.shape) - 1)), keepdim=False) # Take mean over all dims except first
                u.add_(group['eps'])

                m = u.rsqrt() * g
                
                if group['use_rms']:
                    rms_factor = max(1.0, u.square().mean().sqrt())
                    u.div_(rms_factor)
                else:
                    rms_factor = 1.0
                    
                p_norm = p.norm()
                g_norm = g.norm()

                if p_norm != 0. and g_norm != 0.:
                    trust_ratio = (p_norm / g_norm.clamp(min=group['eps2'])).clamp(min=group['min_trust_ratio'])
                    u.mul_(trust_ratio)
                else:
                    trust_ratio = 1.0

                effective_step_size = trust_ratio * torch.rsqrt(state['v'].mean() + group['eps']) / rms_factor
                u.sub_(group['weight_decay'] * effective_step_size * p)

                if group['momentum_beta'] != 0.:
                    state['m'].mul_(group['momentum_beta']).add_(m, alpha=1-group['momentum_beta'])
                    m = state['m']

                p.data.add_(m, alpha=-alpha)
        return loss
