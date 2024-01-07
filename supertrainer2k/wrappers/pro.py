import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import _Wrapper
import torch

class PROWrapper(_Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def rank_loss_inner(logprob_chosen, rank, logprobs, ranks):
            valid = (ranks > rank)
            logprobs_rejected = torch.where(valid, logprobs, float('-inf'))
            logprobs_rejected = torch.where(valid.sum(dim=-1) == 0, 0, logprobs_rejected)
            logdenom = torch.logsumexp(torch.cat((logprob_chosen.unsqueeze(0), logprobs_rejected), dim=0), dim=-1)
            return  -(logprob_chosen - logdenom)
        
        rank_loss_inner_batched = torch.vmap(rank_loss_inner, in_dims=(0, 0, None, None))

        def rank_loss_batch(logprobs, ranks):
            return ((ranks != -100)*rank_loss_inner_batched(logprobs, ranks, logprobs, ranks)).sum() / (ranks != -100).sum()

        self.rank_loss_batched = torch.vmap(rank_loss_batch, in_dims=(0, 0))

    def forward(self, **inputs):
        return self.model(**inputs)

    def lm_loss(self, logits, ranks):
        max_rank, _ = torch.max(ranks)
        logits = logits[ranks == max_rank]
        return logits.mean()

    def get_logits(self, batch, normalize_length = True):
        bsz, comps, seq_len = batch['input_ids'].shape
        
        flat_input_ids = batch['input_ids'].view(-1, seq_len)[:, :-1]
        flat_attn_mask = batch['attention_mask'].view(-1, seq_len)[:, :-1]
        flat_labels = batch['labels'].view(-1, seq_len)[:, 1:]

        flat_attn_mask_e = torch.where(
            (flat_attn_mask.sum(dim=-1) == 0).unsqueeze(-1),
            torch.ones_like(flat_attn_mask),
            flat_attn_mask
        ) # The attention mask for pad sequences is all-zero, which would cause NaN, which annoyed me when debugging

        logits = self.model(input_ids=flat_input_ids, attention_mask=flat_attn_mask_e).logits.log_softmax(dim=-1)

        mask = (flat_labels != -100) * (flat_attn_mask_e != -100)
        logits = torch.gather(logits, -1, torch.where(mask, flat_labels, 0).unsqueeze(-1)).squeeze(-1) * mask
        logits = logits.view(bsz, comps, seq_len - 1).sum(dim=-1)
        
        mask_sum = mask.view(bsz, comps, seq_len - 1).sum(dim=-1)

        if not normalize_length:
            return logits

        return torch.where(mask_sum == 0, 0, logits / mask_sum)

    def training_step(self, batch, batch_idx):
        logprobs = self.get_logits(batch)
        bsz, n_seqs = logprobs.shape

        rank_loss = self.rank_loss_batched(logprobs, batch['ranks']).mean()
        assert not torch.isnan(rank_loss)
        lm_loss = -logprobs[batch['ranks'] == 0].mean()
        assert not torch.isnan(lm_loss)
        loss = lm_loss + rank_loss

        self.log('train/lm_loss', lm_loss)
        self.log('train/rank_loss', rank_loss)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):    
        logits = self.get_logits(batch, normalize_length=False)
        ranks = batch['ranks']
        logits[ranks == -100] = float('-inf')

        _, idxs = torch.max(logits, dim=-1)
        accuracy = (ranks[torch.arange(ranks.shape[0]), idxs] == 0).float().mean()
        self.log('eval/accuracy', accuracy)
        return accuracy
