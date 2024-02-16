import torch
from torch.optim import Optimizer
import warnings
import math

class Adaheavy(Optimizer):
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
        use_rms: bool = True,
        momentum_beta: float = 0.9
    ):
        assert eps >= 0. and eps < 1., "Invalid eps value"
        assert Lambda >= 0. and Lambda <= 1., "Invalid Lambda value"
        assert beta_decay >= 0. and beta_decay <= 1., "Invalid beta_decay value"
        assert momentum_beta >= 0. and momentum_beta <= 1., "Invalid momentum_beta value"

        defaults = dict(
            lr=lr,
            eps=eps,
            eps2=eps,
            min_trust_ratio=min_trust_ratio,
            Lambda=Lambda,
            beta_decay=beta_decay,
            centralize=centralize,
            use_rms=use_rms,
            momentum_beta=momentum_beta
        )

        super(Adaheavy, self).__init__(params, defaults)


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
                    state['v'] = torch.zeros_like(p)
                    state['g_prev'] = torch.zeros_like(p)
                    state['m'] = torch.zeros_like(p)

                state['step'] += 1

                if group['centralize'] and sum(g.shape) > 1:
                    g.sub_(g.mean(dim=tuple(range(1, len(g.shape))), keepdim=True))

                beta1_t = 1.0 - math.pow(state['step'], -group['beta_decay'])

                
                state['m'].mul_(group['momentum_beta']).add_(g, alpha=1-group['momentum_beta'])
                                
                state['v'].mul_(1-beta1_t).add_(g.square(), alpha=beta1_t)
                
                u = state['m'].mul(group['momentum_beta']).add_(g, alpha=1-group['momentum_beta'])

                u.mul_((state['v'] + group['eps']).rsqrt())
                
                u.div_(max(1.0, u.square().mean().sqrt()))

                p_norm = p.norm()
                g_norm = g.norm()

                if p_norm != 0. and g_norm != 0.:
                    u.mul_((p_norm / g_norm.clamp(min=group['eps2'])).clamp(min=group['min_trust_ratio']))
                    u.add_(p - p/p_norm, alpha=group['Lambda'])


                p.data.add_(u, alpha=-alpha)
        return loss
