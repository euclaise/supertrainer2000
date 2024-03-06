import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from .wrapper import Wrapper
import torch
import warnings
from typing import Optional

class LiPOWrapper(Wrapper):
    """
    Lambda loss, inspired by https://arxiv.org/pdf/2402.01878.pdf
    """
    def __init__(self,
        beta: float = 1.0,
        mixce: Optional[float] = None,
        external_ce_labels: bool = False,
        *args,
        **kwargs
    ):
        """
        Args:
            beta (float, optional): Weight of the rank loss in the total loss calculation. Defaults to 1.0.
            mixce (float, Optional): If None, cross-entropy will be used for the LM loss. Otherwise, MixCE will be used with this for beta.
        """
        super().__init__(*args, **kwargs)

        self.beta = beta
        self.mixce = mixce
        self.external_ce_labels = external_ce_labels

        def D(x):
            return torch.log(1 + x.clamp(min=0))

        def rank_loss_pairwise(logprob_chosen, logprob_rejected, rank_chosen, rank_rejected):
            delta = 1/D(rank_chosen) - 1/D(rank_rejected)
            
            return delta * torch.log(1 + torch.exp(-(logprob_chosen - logprob_rejected)))
        rank_loss_pairwise_each = torch.vmap(rank_loss_pairwise, in_dims=(None, None, 0, 0))
            
        def rank_loss_inner(logprob_chosen, rank, logprobs, ranks):
            mask = (ranks > rank)

            pairwise_losses = rank_loss_pairwise_each(logprob_chosen, rank, logprobs, ranks)
            
            return -(pairwise_losses * mask).sum()
        
        rank_loss_inner_batched = torch.vmap(rank_loss_inner, in_dims=(0, 0, None, None))

        def rank_loss_batch(logprobs, ranks, mask):
            valid = mask * (ranks != -100) * (ranks != torch.max(ranks))
            losses = rank_loss_inner_batched(logprobs, ranks, logprobs, ranks)
            
            return (valid*losses).sum() / (valid.sum() + (valid.sum() == 0))

        self.rank_loss_batched = torch.vmap(rank_loss_batch, in_dims=(0, 0, 0))

    def training_step(self, batch, batch_idx):
        ranks = batch['ranks']
        logits, mask = self.get_logits(self.model, batch)
        logprobs = (logits * mask).sum(dim=-1) / (mask.sum(dim=-1) + (mask.sum(dim=-1) == 0))


        if self.external_ce_labels:
            batch_ce = {
                'input_ids': batch['ce_ids'],
                'labels': batch['ce_labels']
            }
            ce_logits, ce_mask = self.get_logits(self.model, batch_ce)
            ce_logprobs = (ce_logits *ce_mask).sum(dim=-1) / (ce_mask.sum(dim=-1) + (ce_mask.sum(dim=-1) == 0))

        if self.mixce is not None:
            lm_losses = -logits if not self.external_ce_labels else -ce_logits
            q = torch.exp(-lm_losses.detach())
            lm_losses = self.mixce * lm_losses + (1 - self.mixce)*q*lm_losses
            lm_losses = (lm_losses * mask).sum(dim=-1) / (mask.sum(dim=-1) + (mask.sum(dim=-1) == 0))
        else:
            lm_losses = -logprobs if not self.external_ce_labels else -ce_logits
            
        mask = mask.sum(dim=-1)
        if self.external_ce_labels:
            lm_loss = (lm_losses * ce_mask).sum() / ce_mask.sum()
        else:
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
