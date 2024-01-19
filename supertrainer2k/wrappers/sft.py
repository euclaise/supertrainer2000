import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import Wrapper
import warnings
import torch
from typing import Optional

class SFTWrapper(Wrapper):
    """
    `SFTWrapper` is a `Wrapper` designed for supervised fine-tuning of transformer models.

    Args:
        mixce (Optional[float]): https://arxiv.org/abs/2305.16958
    """
    def __init__(self,  mixce: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mixce = mixce

    def training_step(self, batch, batch_idx):
        logits, mask = self.get_logits(self.model, batch)

        losses = -logits
        
        if self.mixce is not None:
            q = torch.exp(-losses.detach())
            losses = self.mixce * losses + (1 - self.mixce)*q*losses
            
        loss = (losses * mask).sum() / mask.sum()


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
        with torch.no_grad():
            logits, mask = self.get_logits(self.model, batch)

        loss = -(logits * mask).sum() / mask.sum()
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn(f"NaNs or infs detected in validation. Skipping batch")
            return None
        self.log("validation/loss", loss)
        return {'loss': loss}
