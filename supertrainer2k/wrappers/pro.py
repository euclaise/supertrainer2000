import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import _Wrapper
import torch

class PROWrapper(_Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rank_loss_inner_batched = torch.vmap(self.rank_loss_inner, in_dims=(0, 0, None, None))

        self.rank_loss_batched = torch.vmap(
            lambda logprobs, ranks: rank_loss_inner_batched(logprobs, ranks, logprobs, ranks),
            in_dims=(0, 0)
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    @staticmethod
    def rank_loss_inner(logprob_chosen, rank, logprobs, ranks):
        logprobs_rejected = torch.where(ranks > rank, logprobs, float('-inf'))
        logdenom = torch.logsumexp(logprobs_rejected, dim=-1)
        return -(logprob_chosen - logdenom) * (rank != -100)

    def lm_loss(self, logits, ranks):
        max_rank, _ = torch.max(ranks)
        logits = logits[ranks == max_rank]
        return logits.mean()

    def get_logits(self, batch):
        bsz, comps, seq_len = batch['input_ids'].shape
        
        flat_input_ids = batch['input_ids'].view(-1, seq_len)[:, :-1]
        flat_attn_mask = batch['attention_mask'].view(-1, seq_len)[:, :-1]
        flat_labels = batch['labels'].view(-1, seq_len)[:, 1:]

        logits = self.model(input_ids=flat_input_ids, attention_mask=flat_attn_mask).logits.log_softmax(dim=-1)

        mask = (flat_labels != -100)
        logits = torch.gather(logits, -1, torch.where(mask, flat_labels, 0).unsqueeze(-1)).squeeze(-1) * mask
        logits = logits.view(bsz, comps, seq_len - 1).sum(dim=-1)
        
        mask = mask.view(bsz, comps, seq_len - 1)
        mask_sum = mask.sum(dim=-1)
        mask_zeros = mask_sum == 0

        logits = torch.where(mask_zeros, float('-inf'), logits / mask_sum)

        return logits, mask_sum, mask_zeros

    def training_step(self, batch, batch_idx):
        logits, mask_sum, mask_zeros = self.get_logits(batch)
        logprobs = torch.where(mask_zeros, float('-inf'), logits / mask_sum)
    
        bsz, n_seqs = logprobs.shape

        rank_loss = self.rank_loss_batched(logprobs, batch['ranks']).mean()
        lm_loss = -logits[batch['ranks'] == 0].mean()
        loss = lm_loss + rank_loss

        self.log('train/lm_loss', lm_loss)
        self.log('train/rank_loss', rank_loss)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):    
        logits = self.get_logits(batch, normalize_length=False)
        ranks = batch['ranks']

        _, idxs = torch.max(logits, dim=-1)
        correct = ranks[idxs] == 0

        return correct.float().mean()
