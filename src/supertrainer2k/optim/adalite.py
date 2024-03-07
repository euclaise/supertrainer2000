import torch
from torch.optim import Optimizer
import warnings
import math
import torch.nn.functional as F

class Adalite(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        eps: float = 1e-6,
        beta_m: float = 0.9,
        beta_v: float = 0.999,
        weight_decay: float = 0.01,
        g_norm_min: float = 1e-10,
        ratio_min: float = 1e-4,
        svd_iter: int = 2,
        eps2: float = 1e-10,
        tau: float = 1.
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            beta_m=beta_m,
            beta_v=beta_v,
            weight_decay=weight_decay,
            g_norm_min=g_norm_min,
            ratio_min=ratio_min,
            svd_iter=svd_iter,
            eps2=eps2,
            tau=tau
        )

        super(Adalite, self).__init__(params, defaults)

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

                if len(grad.shape) > 2:
                    grad = grad.reshape(grad.shape[0], -1)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0


                    if len(p.shape) < 2:
                        state['m_avg'] = torch.zeros_like(grad)
                        state['v_avg'] = torch.zeros_like(grad)
                    else:
                        state['v_avg_0'] = torch.zeros_like(grad.mean(dim=1))
                        state['v_avg_1'] = torch.zeros_like(grad.mean(dim=0))
                        
                        state['m_avg_c'] = torch.zeros_like(grad.mean(dim=1)[:, None])
                        state['m_avg_r'] = torch.zeros_like(grad.mean(dim=0)[None, :])
                        state['m_avg_u'] = torch.zeros_like(grad.mean().unsqueeze(0).unsqueeze(0))
\
                state['step'] += 1

                if sum(grad.shape) > 1:
                    trust_ratio = (p.data.norm() / grad.norm().clip(min=group['g_norm_min'])).clip(min=group['ratio_min'])
                    grad.mul_(trust_ratio)



                if len(grad.shape) < 2:
                    m = state['m_avg']
                    v = state['v_avg']
                else:
                    r = state['v_avg_0'][:, None]
                    c = state['v_avg_1'][None, :]
                    v = (r * c) / r.sum().clamp(min=group['eps2'])
                    del r
                    del c
                    m = state['m_avg_c'] @ state['m_avg_u'] @ state['m_avg_r']

                m.lerp_(grad, 1-group['beta_m'])

                v.lerp_((grad - m).square(), 1-group['beta_v'])
                v_avg = v / (1 - group['beta_v'] ** state['step'])
                

                if len(grad.shape) == 2:
                    imp_c = F.softmax(v.mean(dim=1),  dim=0)[:, None]
                    imp_r = F.softmax(v.mean(dim=0), dim=0)[None, :]
                    imp = imp_c * imp_r
                    m.lerp_(grad, 1-imp)
                    
                u = m.lerp(grad, 1-group['beta_m'])

                if len(grad.shape) < 2:
                    state['m_avg'] = m
                    state['v_avg'] = v
                else:
                    state['v_avg_0'] = v.sum(dim=1)
                    state['v_avg_1'] = v.sum(dim=0) / v.sum().clamp(min=group['eps2'])
                        
                    imp_c = F.softmax(v.mean(dim=1) / group['tau'], dim=-1)[:, None]
                    imp_r = F.softmax(v.mean(dim=0) / group['tau'], dim=-1)[None, :]
                    del v

                    C = ((m * imp_r).sum(dim=1))[:, None]
                    R = ((m * imp_c).sum(dim=0))[None, :]

                    s = (C.T @ m @ R.T) / (C.T @ C @ R @ R.T).clamp(min=group['eps2'])

                    state['m_avg_c'] = C
                    state['m_avg_r'] = R
                    state['m_avg_u'] = s
                del m
 

                u.div_((v_avg + group['eps']).sqrt())
                
                u = u.reshape(p.data.shape)
                u.add_(p.data, alpha=group['weight_decay'])


                p.data.add_(u, alpha=-group['lr'])

        return loss
