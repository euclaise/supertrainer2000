import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from .wrapper import Wrapper
import torch
import copy
import warnings
from typing import Optional, Literal

class PairwiseWrapper(Wrapper): # https://arxiv.org/abs/2401.08417
    """
        Wrapper for PRO/RRHF/unlikelihood
    """
    def __init__(
        self,
        beta: float = 0.1, 
        eps: float = 0.0,
        method: Literal["rrhf", "cpo"] = "rrhf",
        normalize_length: bool = True,
        mixce: Optional[float] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.eps = eps
        self.mixce = mixce
        self.normalize_length = normalize_length
        self.method = method

        if self.method == "cpo":
            def pairwise_loss(lp_chosen, lp_rejected):
                h = lp_chosen - lp_rejected
                return -(1-self.eps)*F.logsigmoid(h) + -self.eps*F.logsigmoid(-h)
        elif self.method == "rrhf":
            def pairwise_loss(lp_chosen, lp_rejected):
                h = lp_rejected - lp_chosen
                return h.clamp(min=0)
        elif self.method == "unlikelihood":
            def pairwise_loss(lp_chosen, lp_rejected):
                unlik = torch.log(1 - torch.exp(lp_rejected))
                return -(lp_chosen + unlik)
        else:
            raise ValueError(f"Unsupported pairwise comparison method: {method}")

        multi_pairwise_loss = torch.vmap(pairwise_loss, in_dims=(None, 0)) # vmap over multiple rejections
        
        def listwise_loss(lp_chosen, rank_chosen, lps, ranks):
            valid_comps = ranks > rank_chosen
            comps = multi_pairwise_loss(lp_chosen, lps) * valid_comps
            return comps.sum() / (valid_comps.sum() + (valid_comps.sum() == 0))

        multi_listwise_loss = torch.vmap(listwise_loss, in_dims=(0, 0, None, None))
        def batchwise_loss(lps, ranks, mask):
            valid_comps = (ranks != -100) * (ranks != torch.max(ranks)) * mask
            comps = multi_listwise_loss(lps, ranks, lps, ranks) * valid_comps
            return comps.sum() / (valid_comps.sum() + (valid_comps.sum() == 0))

        self.rank_loss_batched = torch.vmap(batchwise_loss, in_dims=(0, 0, 0))

    def training_step(self, batch, batch_idx):
        ranks = batch['ranks']
        logits, mask = self.get_logits(self.model, batch)
        logprobs = (logits * mask).sum(dim=-1) / (mask.sum(dim=-1) + (mask.sum(dim=-1) == 0))

        if self.mixce is not None:
            lm_losses = -logits
            q = torch.exp(-lm_losses.detach())
            lm_losses = self.mixce * lm_losses + (1 - self.mixce)*q*lm_losses
            lm_losses = (lm_losses * mask).sum(dim=-1) / (mask.sum(dim=-1) + (mask.sum(dim=-1) == 0))
        else:
            lm_losses = -logprobs

        if not self.normalize_length:
            logprobs = (logits * mask).sum(dim=-1)
            
        mask = mask.sum(dim=-1)
        lm_loss = (lm_losses[ranks == 0] * mask[ranks == 0]).sum() / mask[ranks == 0].sum()

            

        bsz, n_seqs = logprobs.shape
        rank_loss = self.rank_loss_batched(logprobs, ranks, mask)

        rank_loss = rank_loss.mean()
        loss = lm_loss + self.beta*rank_loss

        if torch.isnan(loss):
            self.nan_counter += 1
            self.consecutive_nans += 1
            assert self.consecutive_nans <= self.skip_nans
            warnings.warn(f"NaNs or infs detected ({self.nan_counter} in training so far). Skipping batch.")
            return None
        self.consecutive_nans = 0

        self.log('train/lm_loss', lm_loss)
        self.log('train/rank_loss', rank_loss)
        self.log('train/loss', loss)

        self.ema_step()

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            logits, mask = self.get_logits(self.model, batch)
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
