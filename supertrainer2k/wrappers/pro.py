import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import Wrapper
import torch
import warnings

class PROWrapper(Wrapper):
    """
    `PROWrapper` is a `Wrapper` designed for training transformer models using a length-normalized cross-entropy loss over ranking data.

    This method was introduced by Song et al. in https://arxiv.org/abs/2306.17492
    """
    def __init__(self, beta: float = 1.0, use_average: bool = True, *args, **kwargs):
        """
        Args:
            beta (float, optional): Weight of the rank loss in the total loss calculation. Defaults to 1.0.
            use_average (bool, optional): Flag to determine if averaging is used in rank loss calculation instead of summation. This may help reduce loss instability when the number of response options varies. Defaults to True.
        """
        super().__init__(*args, **kwargs)

        self.use_average = use_average
        self.beta = beta
            
        def rank_loss_inner(logprob_chosen, rank, logprobs, ranks):
            mask = (ranks > rank)

            all_max = torch.where(mask, logprobs, logprobs.min()).max(dim=-1, keepdim=True)[0]
            all_max = torch.maximum(all_max, logprob_chosen)
            all_exp = torch.exp(logprobs - all_max) * mask
            p_chosen = torch.exp(logprob_chosen - all_max)

            if self.use_average:
                denom = p_chosen + (all_exp.sum(dim=-1) / (mask.sum(dim=-1) + (mask.sum(dim=-1) == 0)))
            else:
                denom = p_chosen + all_exp.sum(dim=-1)

            return -torch.log(p_chosen / denom)
        
        rank_loss_inner_batched = torch.vmap(rank_loss_inner, in_dims=(0, 0, None, None))

        def rank_loss_batch(logprobs, ranks, mask):
            valid = mask * (ranks != -100) * (ranks != torch.max(ranks))
            losses = rank_loss_inner_batched(logprobs, ranks, logprobs, ranks)
            
            return (valid*losses).sum() / valid.sum()

        self.rank_loss_batched = torch.vmap(rank_loss_batch, in_dims=(0, 0, 0))

    def training_step(self, batch, batch_idx):
        ranks = batch['ranks']
        logits, mask = self.get_logits(self.model, batch)
        logprobs = (logits * mask).sum(dim=-1)

        bsz, n_seqs = logprobs.shape
        rank_loss = self.rank_loss_batched(logprobs, ranks, mask)

        rank_loss = rank_loss.mean()
        lm_loss = -(logprobs[ranks == 0] * mask[ranks == 0]).sum() / mask[ranks == 0].sum()
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
