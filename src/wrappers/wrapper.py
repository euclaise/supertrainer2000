import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim import lr_scheduler
from typing import Optional, Dict, Literal, Callable
from ..datasets import DataModule

class _Wrapper(L.LightningModule):
    def __init__(self,
        model: transformers.PreTrainedModel,
        data_module: DataModule,
        optimizer: Optimizer,
        scheduler: Callable,
        scheduler_config: Optional[Dict] = None,
        optim_params: Optional[Dict] = {}
    ):
        super().__init__()
        self.model = model
        self.data_module = data_module
        self.optimizer_cls = optimizer
        self.scheduler_fn = scheduler
        self.scheduler_step = scheduler_step
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
        optim = self.optimizer_cls(params = self.parameters(), lr=self.lr)


        total_steps = self.trainer.estimated_stepping_batches if scheduler['interval'] == 'step' else self.trainer.max_epochs
    
        scheduler['scheduler'] = self.scheduler(optim, total_steps=total_steps)

        return [self.optimizer], [self.scheduler]

    def save(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        self.model.push_to_hub(*args, **kwargs)
