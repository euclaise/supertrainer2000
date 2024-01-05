import torch
from torch.optim import Optimizer


# Dummy optimizer for use with apply_optimizer_in_backward

class _DummyOptimizer(Optimizer):
    def __init__(
        self,
        params
    ):
        super(_DummyOptimizer, self).__init__(params, {'lr': 0.0})


    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        return loss

    def zero_grad(self):
        pass
