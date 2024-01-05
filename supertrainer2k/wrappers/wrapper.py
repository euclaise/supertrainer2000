import lightning as L
import transformers
from torch.optim import Optimizer,  lr_scheduler
from typing import Optional, Dict, Literal, Callable
from ..datasets import DataModule
from ..optim.dummy import _DummyOptimizer
from torch.distributed.optim.apply_optimizer_in_backward import _apply_optimizer_in_backward

class _Wrapper(L.LightningModule):
    def __init__(self,
        model: transformers.PreTrainedModel,
        optimizer: Optimizer,
        scheduler: Callable,
        lr: float,
        adalite_backward: bool = False,
        scheduler_config: Optional[Dict] = None,
        optimizer_params: Optional[Dict] = {}
    ):
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer
        self.scheduler = scheduler
        self.optimizer_params = optimizer_params
        self.lr = lr
        self.adalite_backward = adalite_backward

        if scheduler_config is not None:
            self.scheduler_config = scheduler_config
        else:
            self.scheduler_config = {
                'interval': 'step',
                'frequency': 1
            }
    
    def configure_optimizers(self):
        if self.adalite_backward:
            if self.trainer.accumulate_grad_batches > 1:
                raise ValueError("Gradient accumulation is incompatible with Adalite-style backwarsds passes. Please pass adalite_backward=False to your Wrapper, or do not use gradient accumulation.")

            self.optimizer_params['lr'] = self.lr
            _apply_optimizer_in_backward(
                self.optimizer_cls,
                self.model.parameters(),
                optimizer_kwargs = self.optimizer_params
            )

            self.optimizer = _DummyOptimizer(self.model.parameters())
        else:  
            self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr, **self.optimizer_params)

        scheduler = self.scheduler_config
        total_steps = self.trainer.estimated_stepping_batches if scheduler['interval'] == 'step' else self.trainer.max_epochs

        scheduler['scheduler'] = self.scheduler(self.optimizer, total_steps=total_steps)


        return [self.optimizer], [scheduler]

    def save(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        self.model.push_to_hub(*args, **kwargs)
