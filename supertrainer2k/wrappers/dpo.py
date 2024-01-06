import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from .wrapper import _Wrapper
import torch
import copy

class DPOWrapper(_Wrapper):
    def __init__(self, ref_model: torch.nn.Module, beta: float = 0.1, eps: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.eps = eps
        self.ref_model = copy.deepcopy(ref_model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        def pairwise_loss(lp_cur_chosen, lp_cur_rejected, lp_ref_chosen, lp_ref_rejected):
            pol_logratio = lp_cur_chosen - lp_cur_rejected
            ref_logratio = lp_ref_chosen - lp_ref_rejected

            h = pol_logratio - ref_logratio
            return -F.logsigmoid(self.beta * h) * (1 - self.eps) - F.logsigmoid(-self.beta * h)*self.eps

        f1 = torch.vmap(pairwise_loss, in_dims=(None, 0, None, 0)) # vmap over multiple rejects
        
        def listwise_loss(lp_cur_chosen, lp_ref_chosen, rank_chosen, lps_cur, lps_ref, ranks):
            valid_comps = (ranks > rank_chosen).float()
            comps = f1(lp_cur_chosen, lps_cur, lp_ref_chosen, lps_ref) * valid_comps
            return comps.sum() / valid_comps.sum()

        f2 = torch.vmap(listwise_loss, in_dims=(0, 0, 0, None, None, None))
        def batchwise_loss(lps_cur, lps_ref, ranks):
            valid_comps = (ranks != -100).float() * (ranks != -100)
            comps = f2(lps_cur, lps_ref, ranks, lps_cur, lps_ref, ranks) * valid_comps
            return comps.sum() / valid_comps.sum()

        batched_loss = torch.vmap(batchwise_loss, in_dims=(0, 0, 0))
        self.dpo_loss = lambda lps_cur, lps_ref, ranks: batched_loss(lps_cur, lps_ref, ranks).sum() / lps_cur.shape[0]

    def forward(self, **inputs):
        return self.model(**inputs)

    def get_logits(self, model, batch):
        bsz, comps, seq_len = batch['input_ids'].shape
        
        flat_input_ids = batch['input_ids'].view(-1, seq_len)[:, :-1]
        flat_labels = batch['labels'].view(-1, seq_len)[:, 1:]

        logits = model(input_ids=flat_input_ids).logits.log_softmax(dim=-1)

        mask = (flat_labels != -100)
        logits = torch.gather(logits, -1, torch.where(mask, flat_labels, 0).unsqueeze(-1)).squeeze(-1) * mask
        logits = logits.view(bsz, comps, seq_len - 1).sum(dim=-1)
        
        mask = mask.view(bsz, comps, seq_len - 1)
        mask_sum = mask.sum(dim=-1)
        mask_zeros = mask_sum == 0
        mask_sum = torch.where(mask_zeros, 1, mask_sum)

        return logits 

    def training_step(self, batch, batch_idx):
        ranks = batch['ranks']
        
        lp_cur = self.get_logits(self.model, batch)
        assert lp_cur.shape == ranks.shape
        with torch.no_grad():
            lp_ref = self.get_logits(self.ref_model, batch)

        loss = self.dpo_loss(lp_cur, lp_ref, ranks)

        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self.get_logits(batch, normalize_length=False)

        # Logits is of shape [bsz, comps]
        idxs = torch.argmax(logits)

        return (idxs == 0).float().mean()
