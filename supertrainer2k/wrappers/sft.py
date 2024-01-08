import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import Wrapper
import warnings
import torch

class SFTWrapper(Wrapper):
    """
    `SFTWrapper` is a `Wrapper` designed for supervised fine-tuning of transformer models.
    """

    def training_step(self, batch, batch_idx):
        logits, mask = self.get_logits(self.model, batch)
  
        loss = -(logits * mask).sum() / mask.sum()
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_counter += 1
            self.consecutive_nans += 1
            assert self.consecutive_nans <= self.skip_nans
            warnings.warn(f"NaNs or infs detected ({self.nan_counter} in training so far). Skipping batch")
            return None
        self.consecutive_nans = 0
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits, mask = self.get_logits(self.model, batch, normalize_length=False)


        loss = -(logits * mask).sum() / mask.sum()
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn(f"NaNs or infs detected in validation. Skipping batch")
            return None
        self.log("validation/loss", loss)
        return {'loss': loss}
