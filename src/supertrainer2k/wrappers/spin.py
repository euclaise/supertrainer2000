import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import Wrapper
import torch
import warnings
import torch.nn as nn

class SPINWrapper(Wrapper):
    """
    `SPINWrapper` is a wrapper for training with self-play finetuning, see https://arxiv.org/abs/2401.01335 for details

    Note that `SPINWrapper` only performs one SPIN iteration, the generation part must be handled by the user.
    """
    def __init__(self, ref_model: nn.Module, beta: float = 1.0, *args, **kwargs):
        """
        Args:
            beta (float): Regluarization parameter. The SPIN paper refers to it as lambda.
            ref_model (nn.Module): Reference model - i.e. the model from phe previous SPIN iteration.
        """
        super().__init__(*args, **kwargs)

        self.use_average = use_average
        self.beta = beta
        self.ref = ref_model
        for p in self.ref.parameters():
            p.requires_grad = False

    def training_step(self, batch, batch_idx):
        ranks = batch['ranks']
        logits, mask = self.get_logits(self.model, batch)
        with torch.no_grad():
            logits_ref, _ = self.get_logits(self.ref_model, batch)
        logprobs = (logits * mask).sum(dim=-1) # [bsz, num_seq]
        logprobs_ref = (logits_ref * mask).sum(dim=-1)

        ratio_l = logprobs_ref[ranks == 0] - logprobs_ref[ranks == 0]
        ratio_r = logprobs_ref[ranks == 1] - logprobs_ref[ranks == 1]
        ratio = (ratio_l - ratio_r)
        loss = torch.log(1 + torch.exp(-self.beta * ratio))

        if torch.isnan(loss):
            self.nan_counter += 1
            self.consecutive_nans += 1
            assert self.consecutive_nans <= self.skip_nans
            warnings.warn(f"NaNs or infs detected ({self.nan_counter} in training so far). Skipping batch.")
            return None
        self.consecutive_nans = 0

        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits, mask = self.get_logits(self.model, batch, normalize_length=False)
        logits = (logits*mask).sum(dim=-1)

        ranks = batch['ranks']
        logits[ranks == -100] = float('-inf')

        _, idxs = torch.max(logits, dim=-1)
        accuracy = (ranks[torch.arange(ranks.shape[0]), idxs] == 0).float().mean()

        if torch.isnan(accuracy) or torch.isinf(accuracy):
            warnings.warn(f"NaNs or infs detected in validation. Skipping batch.")
            return None
        self.log('eval/accuracy', accuracy)
        return accuracy
