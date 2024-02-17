import torch
from torch.optim import Optimizer
import warnings
import math

class Lilith(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        eps: float = 1e-8,
        beta1_m: float = 0.9,
        beta2_m: float = 0.9,
        beta_v: float = 0.999,
        weight_decay: float = 0.01,
        m_norm_min: float = 1e-4,
        ratio_min: float = 1e-4,
        lookahead_k: int = 5,
        lookahead_beta: float = 0.5
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            beta1_m=beta1_m,
            beta2_m=beta2_m,
            beta_v=beta_v,
            weight_decay=weight_decay,
            m_norm_min=m_norm_min,
            ratio_min=ratio_min,
            lookahead_k = lookahead_k,
            lookahead_beta=lookahead_beta
        )

        super(Lilith, self).__init__(params, defaults)

    @torch.no_grad
    def step(self, closure):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m_avg1'] = torch.zeros_like(grad)
                    state['m_avg2'] = torch.zeros_like(grad)
                    state['v_avg'] = torch.zeros_like(grad)
                    state['ema'] = p.data.clone()

                state['step'] += 1

                

                if sum(grad.shape) > 1:
                    trust_ratio = (p.data.norm() / grad.norm().clip(min=1e-4)).clip(min=group['ratio_min'])
                    grad.sub_(grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True))
                    grad.mul_(trust_ratio)

                m_avg1_prev = state['m_avg1'].clone()
                state['m_avg1'].add_(state['m_avg2']).lerp_(grad, 1-group['beta1_m'])
                state['m_avg2'].lerp_(state['m_avg1'] - m_avg1_prev, 1-group['beta2_m'])

                
                u = state['m_avg1'] + state['m_avg2']

                
                state['v_avg'].lerp_(u.square(), 1-group['beta_v'])

                u.div_(state['v_avg'].sqrt() + group['eps'])

                u.add_(p, alpha=group['weight_decay'])

                p.data.add_(u, alpha=-group['lr'])

                if state['step'] % group['lookahead_k'] == 0:
                    state['ema'].lerp_(p.data, 1-group['lookahead_beta'])
                    p.data.copy_(state['ema'])

        return loss
