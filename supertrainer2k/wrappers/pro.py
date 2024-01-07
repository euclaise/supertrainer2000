import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import Wrapper
import torch
import warnings

class PROWrapper(Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
        def rank_loss_inner(logprob_chosen, rank, logprobs, ranks):
            mask = (ranks > rank)

            all_max = torch.where(mask, logprobs, logprobs.min()).max(dim=-1, keepdim=True)[0]
            all_max = torch.maximum(all_max, logprob_chosen)
            all_exp = torch.exp(logprobs - all_max) * mask
            p_chosen = torch.exp(logprob_chosen)
            
            return -(torch.log(p_chosen /(p_chosen + all_exp[1:].sum(dim=-1))))
        
        rank_loss_inner_batched = torch.vmap(rank_loss_inner, in_dims=(0, 0, None, None))

        def rank_loss_batch(logprobs, ranks, mask):
            valid = mask * (ranks != -100)
            return (valid*rank_loss_inner_batched(logprobs, ranks, logprobs, ranks)).sum() / valid.sum()

        self.rank_loss_batched = torch.vmap(rank_loss_batch, in_dims=(0, 0, 0))

    def training_step(self, batch, batch_idx):
        ranks = batch['ranks']
        try:
            logprobs, mask = self.get_logits(self.model, batch)
        except AssertionError as e:
            self.nan_counter += 1
            self.consecutive_nans += 1
            assert self.consecutive_nans <= self.skip_nans
            warnings.warn(f"NaNs detected ({self.nan_counter} in training so far). Skipping batch.")
            return None
        self.consecutive_nans = 0
        bsz, n_seqs = logprobs.shape
    
        rank_loss = self.rank_loss_batched(torch.where(logprobs == 0, logprobs, 1e-6), ranks, mask).mean()
        assert not torch.isinf(rank_loss) and not torch.isnan(rank_loss), rank_loss
        lm_loss = -(logprobs[ranks == 0] * mask[ranks == 0]).sum() / mask[ranks == 0].sum()
        assert not torch.isinf(lm_loss) and not torch.isnan(lm_loss), lm_loss
        loss = lm_loss + rank_loss

        self.log('train/lm_loss', lm_loss)
        self.log('train/rank_loss', rank_loss)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        try:   
            logits, mask = self.get_logits(self.model, batch, normalize_length=False)
        except AssertionError as e:
            warnings.warn(f"NaNs detected in validation. Skipping batch.")
            return None
        ranks = batch['ranks']
        logits[ranks == -100] = float('-inf')
        logits[mask] = float('-inf')

        _, idxs = torch.max(logits, dim=-1)
        accuracy = (ranks[torch.arange(ranks.shape[0]), idxs] == 0).float().mean()
        self.log('eval/accuracy', accuracy)
        return accuracy
