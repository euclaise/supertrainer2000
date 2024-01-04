import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class SFTWrapper(L.LightningModule):
    def __init__(self,
        model: transformers.PreTrainedModel,
        data_module: L.LightningDataModule,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
    ):
        super().__init__()
        self.model = model
        self.data_module = data_module
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        return self(**batch).loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return {'loss': self(**batch).loss}

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
