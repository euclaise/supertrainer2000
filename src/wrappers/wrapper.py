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
        scheduler_config: Optional[Dict] = None,
        optim_params: Optional[Dict] = {}
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optim_params = optim_params

        if scheduler_config is not None:
            self.scheduler_config = scheduler_config
        else:
            self.scheduler_config = {
                'interval': 'step',
                'frequency': 1
            }
    
    def configure_optimizers(self):
        scheduler = self.scheduler_config
        total_steps = self.trainer.estimated_stepping_batches if scheduler['interval'] == 'step' else self.trainer.max_epochs

        scheduler['scheduler'] = self.scheduler(self.optimizer, total_steps=total_steps)

        return [self.optimizer], [scheduler]

    def save(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        self.model.push_to_hub(*args, **kwargs)
