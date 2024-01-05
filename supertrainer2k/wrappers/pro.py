import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .wrapper import _Wrapper
import torch

class PROWrapper(_Wrapper):
    def forward(self, **inputs):
        return self.model(**inputs)

    def pro_loss(self, logprobs, mask):
        logprobs = torch.where(mask, logprobs, float('-inf'))
        logdenom = torch.logsumexp(logprobs, dim=-1)
        lm_loss = -logprobs[:, 0]
        rank_loss = -(logprobs[:, 0] - logdenom)
        return lm_loss, rank_loss

    def get_logits(self, batch, normalize_length = True):
        bsz, comps, seq_len = batch['input_ids'].shape
        
        flat_input_ids = batch['input_ids'].view(-1, seq_len)[:, :-1]
        flat_labels = batch['labels'].view(-1, seq_len)[:, 1:]

        logits = self.model(input_ids=flat_input_ids).logits.log_softmax(dim=-1)

        mask = (flat_labels != -100)
        logits = torch.gather(logits, -1, torch.where(mask, flat_labels, 0).unsqueeze(-1)).squeeze(-1) * mask
        logits = logits.view(bsz, comps, seq_len - 1).sum(dim=-1)
        
        mask = mask.view(bsz, comps, seq_len - 1)
        mask_sum = mask.sum(dim=-1)
        mask_zeros = mask_sum == 0
        mask_sum = torch.where(mask_zeros, 1, mask_sum)

        if normalize_length:
            logits = torch.where(mask_zeros, float('-inf'), logits / mask_sum)

        return logits 

    def training_step(self, batch, batch_idx):
        lp_cur = self.get_logits(batch)
        
        lm_loss, rank_loss = self.pro_loss(lp_cur, batch['mask'])
        
        lm_loss = lm_loss.mean()
        rank_loss = rank_loss.mean()
        loss = lm_loss + rank_loss

        self.log('lm_loss', lm_loss)
        self.log('rank_loss', rank_loss)
        self.log('loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):    
        logits = self.get_logits(batch, normalize_length=False)

        # Logits is of shape [bsz, comps]
        idxs = torch.argmax(logits)

        return (idxs == 0).float().mean()
