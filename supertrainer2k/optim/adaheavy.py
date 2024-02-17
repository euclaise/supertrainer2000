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
        m_beta1: float = 0.9,
        m_beta2: float = 0.99,
        n: float = 1,
        k: int = 5,
        ema_beta: float = 0.5,
        lookahead: bool = True
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            eps2=eps2,
            min_trust_ratio=min_trust_ratio,
            Lambda=Lambda,
            beta_decay=beta_decay,
            centralize=centralize,
            use_rms=use_rms,
            m_beta1=m_beta1,
            m_beta2=m_beta2,
            n=n,
            k=k,
            ema_beta=ema_beta,
            lookahead=lookahead
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
                    state['m1'] = torch.zeros_like(p)
                    state['m2'] = torch.zeros_like(p)
                    if group['lookahead']:
                        state['ema'] = p.data.clone()
                    
                state['step'] += 1

                if group['centralize'] and sum(g.shape) > 1:
                    g.sub_(g.mean(dim=tuple(range(1, len(g.shape))), keepdim=True))

                beta1_t = 1.0 - math.pow(state['step'], group['beta_decay'])

                
                # m1 = beta*(m1 + m2) + (1-beta)*u
                # m2 = beta*m2 + (1-beta)*(m1 - m1_pre)
                m1_new = state['m1'].add(state['m2']).mul_(group['m_beta1']).add_(g, alpha=1-group['m_beta1'])
                state['m2'].mul_(group['m_beta2']).add_(m1_new, alpha=1-group['m_beta2']).sub_(state['m1'], alpha=1-group['m_beta2'])
                state['m1'].copy_(m1_new)
                u = state['m1'].add(state['m2'], alpha=group['n'])

                state['v'].mul_(beta1_t).add_(u.square(), alpha=1-beta1_t) # Momentum-in-momentum https://openreview.net/forum?id=qQz1UKDCiy7


                u.mul_(state['v'].add(group['eps']).rsqrt())

                p_norm = p.norm()
                g_norm = g.norm()

                if p_norm != 0 and g_norm != 0:
                    trust_ratio = (p_norm / g_norm.clamp(min=group['eps2'])).clamp(min=group['min_trust_ratio'])
                    u.mul_(trust_ratio)
                else:
                    trust_ratio = 1

                u.add_(p.data, alpha=group['Lambda'])
                # LAMB scales the weight decay by trust ratio
                # However, to match adaptive weight decay, I remove this scaling
                # See: https://proceedings.neurips.cc/paper_files/paper/2023/file/f9d7d6c695bc983fcfb5b70a5fbdfd2f-Paper-Conference.pdf

                p.data.add_(u, alpha=-alpha)

                if group['lookahead'] and (state['step'] + 1) % group['k'] == 0:
                    state['ema'].mul_(group['ema_beta']).add_(p, alpha=1-group['ema_beta'])
                    p.data.copy_(state['ema'])
        return loss
