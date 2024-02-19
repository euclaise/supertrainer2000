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
        beta2_m: float = 0.99,
        beta_v: float = 0.999,
        weight_decay: float = 0.01,
        g_norm_min: float = 1e-4,
        ratio_min: float = 1e-4,
        acceleration: float = 1,
        ema_k: int = 0,
        ema_beta: float = 0.99
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            beta1_m=beta1_m,
            beta2_m=beta2_m,
            beta_v=beta_v,
            weight_decay=weight_decay,
            g_norm_min=g_norm_min,
            ratio_min=ratio_min,
            acceleration=acceleration,
            ema_k = ema_k,
            ema_beta=ema_beta,
        )

        super(Lilith, self).__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
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
                    if group['ema_k'] > 0:
                        state['ema'] = p.data.clone()

                state['step'] += 1

                if sum(grad.shape) > 1:
                    trust_ratio = (p.data.norm() / grad.norm().clip(min=group['g_norm_min'])).clip(min=group['ratio_min'])
                    grad.mul_(trust_ratio)

                m_avg1_prev = state['m_avg1'].clone()
                state['m_avg1'].add_(state['m_avg2']).lerp_(grad, 1-group['beta1_m'])
                state['m_avg2'].lerp_(state['m_avg1'] - m_avg1_prev, 1-group['beta2_m'])

                u = state['m_avg1'] + group['acceleration']*state['m_avg2']

                state['v_avg'].lerp_(u.square(), 1-group['beta_v'])
                v_avg = state['v_avg'] / (1 - group['beta_v'] ** state['step'])

                u.div_((state['v_avg'] + group['eps']).sqrt())

                u.add_(p, alpha=group['weight_decay'])

                p.data.add_(u, alpha=-group['lr'])

                if group['ema_k'] != 0:
                    state['ema'].lerp_(p.data, group['ema_beta'])
                    if state['step'] % group['ema_k'] == 0:
                        p.data.copy_(state['ema'])

        return loss
