import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import _Wrapper

class SFTWrapper(_Wrapper):
    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        return self(**batch).loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return {'loss': self(**batch).loss}
