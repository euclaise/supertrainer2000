import lightning as L
import transformers
from torch.optim import Optimizer,  lr_scheduler
from typing import Optional, Dict, Literal, Callable
from ..datasets import DataModule
    

class _Wrapper(L.LightningModule):
    def __init__(self,
        model: transformers.PreTrainedModel,
        batch_size: int,
        optimizer: Optimizer,
        scheduler: Callable,
        lr: float,
        scheduler_config: Optional[Dict] = None,
        optimizer_params: Optional[Dict] = {}
    ):
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer
        self.scheduler = scheduler
        self.optimizer_params = optimizer_params
        self.lr = lr

        if scheduler_config is not None:
            self.scheduler_config = scheduler_config
        else:
            self.scheduler_config = {
                'interval': 'step',
                'frequency': 1
            }
    
    def configure_optimizers(self):
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr, **self.optimizer_params)
        scheduler = self.scheduler_config

        total_steps = self.trainer.estimated_stepping_batches if scheduler['interval'] == 'step' else self.trainer.max_epochs

        scheduler['scheduler'] = self.scheduler(self.optimizer, total_steps=total_steps)

        return [self.optimizer], [scheduler]

    def save(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        self.model.push_to_hub(*args, **kwargs)
