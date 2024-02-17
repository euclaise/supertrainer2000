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
        k: float = 2
    ):
        assert eps >= 0. and eps < 1., "Invalid eps value"
        assert Lambda >= 0. and Lambda <= 1., "Invalid Lambda value"
        assert beta_decay >= 0. and beta_decay <= 1., "Invalid beta_decay value"

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
            k=k
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
                    state['e_max'] = torch.zeros_like(p.mean())

                state['step'] += 1

                if group['centralize'] and sum(g.shape) > 1:
                    g.sub_(g.mean(dim=tuple(range(1, len(g.shape))), keepdim=True))

                beta1_t = 1.0 - math.pow(state['step'], -group['beta_decay'])

                # Second moment
                state['v'].mul_(1-beta1_t).add_(g.square(), alpha=beta1_t)

                u = g.mul(state['v'].add(group['eps']).rsqrt())

                if group['use_rms']:
                    u.div_(max(1.0, u.square().mean().sqrt()))

                p_norm = p.norm()
                g_norm = g.norm()

                if p_norm != 0. and g_norm != 0.:
                    u.mul_((p_norm / g_norm.clamp(min=group['eps2'])).clamp(min=group['min_trust_ratio']))
                    u.add_(p - p/p_norm, alpha=group['Lambda'])

                
                # Double EMA, computed on updates as per https://arxiv.org/pdf/2002.04839.pdf
                # m1 = beta*(m1 + m2) + (1-beta)*u
                # m2 = beta*m2 + (1-beta)*(m1 - m1_pre)
                m1_prev = state['m1'].clone()
                state['m1'].add_(state['m2']).mul_(group['m_beta1']).add_(u, alpha=1-group['m_beta1'])
                state['m2'].mul_(group['m_beta2']).add_(state['m1'], alpha=1-group['m_beta2']).sub_(m1_prev, alpha=1-group['m_beta2'])
                                
                u = state['m1'].add(state['m2'], alpha=group['k'])
                #u.div_(1 - (group['m_beta1'] * group['m_beta2'])**state['step']) # bias correction, may not be needed, see https://arxiv.org/pdf/2110.10828.pdf


                p.data.add_(u, alpha=-alpha)
        return loss
