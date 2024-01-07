import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import Wrapper
import torch
import warnings

class PROWrapper(Wrapper):
    def __init__(self, use_average=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_average = use_average
            
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
        try:
            logprobs, mask = self.get_logits(self.model, batch)
        except AssertionError as e:
            self.nan_counter += 1
            self.consecutive_nans += 1
            assert self.consecutive_nans <= self.skip_nans
            warnings.warn(f"NaNs or infs detected ({self.nan_counter} in training so far). Skipping batch.")
            return None
        self.consecutive_nans = 0
        bsz, n_seqs = logprobs.shape
    
        rank_loss = self.rank_loss_batched(logprobs, ranks, mask).mean()        
        lm_loss = -(logprobs[ranks == 0] * mask[ranks == 0]).sum() / mask[ranks == 0].sum()
        
        loss = lm_loss + rank_loss

        self.log('train/lm_loss', lm_loss)
        self.log('train/rank_loss', rank_loss)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        try:   
            logits, mask = self.get_logits(self.model, batch, normalize_length=False)
        except AssertionError as e:
            warnings.warn(f"NaNs or infs detected in validation. Skipping batch.")
            return None
        ranks = batch['ranks']
        logits[ranks == -100] = float('-inf')
        logits[mask] = float('-inf')

        _, idxs = torch.max(logits, dim=-1)
        accuracy = (ranks[torch.arange(ranks.shape[0]), idxs] == 0).float().mean()
        self.log('eval/accuracy', accuracy)
        return accuracy
