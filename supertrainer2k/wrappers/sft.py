import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import Wrapper
import warnings

class SFTWrapper(Wrapper):
    def training_step(self, batch, batch_idx):
        try:
            logits, mask = self.get_logits(self.model, batch)
        except AssertionError as e:
            self.nan_counter += 1
            self.consecutive_nans += 1
            assert self.consecutive_nans <= self.skip_nans
            warnings.warn(f"NaNs detected ({self.nan_counter} in training so far). Skipping batch")
            return None

        self.consecutive_nans = 0
        loss = -(logits * mask).sum() / mask.sum()

        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            logits, mask = self.get_logits(self.model, batch, normalize_length=False)
        except AssertionError as e:
            warnings.warn(f"NaNs detected in validation. Skipping batch")
            return None

        loss = -(logits * mask).sum() / mask.sum()

        self.log("validation/loss", loss)
        return {'loss': loss}
