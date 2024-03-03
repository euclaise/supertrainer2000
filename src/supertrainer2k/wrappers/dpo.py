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

class DPOWrapper(Wrapper):
    """
    Unfinished.
    """
    def __init__(
        self,
        ref_model: Optional[torch.nn.Module],
        beta: float = 0.1, 
        method = Literal["dpo", "cdpo", "ipo", "rso"],
        eps: Optional[float] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.eps = eps if method != "dpo" else 0.0
        
        self.ref_model = copy.deepcopy(ref_model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        match method:
            case "dpo" | "cdpo":
                def pairwise_loss(lp_cur_chosen, lp_cur_rejected, lp_ref_chosen, lp_ref_rejected):
                    chosen_logratio = self.beta*(lp_cur_chosen - lp_ref_chosen)
                    rejected_logratio = self.beta*(lp_cur_rejected - lp_ref_rejected)

                    h = chosen_logratio - rejected_logratio
                    return -F.logsigmoid(h) * (1 - self.eps) - F.logsigmoid(-h)*self.eps
            case "ipo":
                def pairwise_loss(lp_cur_chosen, lp_cur_rejected, lp_ref_chosen, lp_ref_rejected):
                    chosen_logratio = lp_cur_chosen - lp_ref_chosen
                    rejected_logratio = lp_cur_rejected - lp_ref_rejected

                    h = chosen_logratio - rejected_logratio
                    return (h - 1/(2 * self.beta)) ** 2
            case "rso":
                def pairwise_loss(lp_cur_chosen, lp_cur_rejected, lp_ref_chosen, lp_ref_rejected):
                    chosen_logratio = self.beta*(lp_cur_chosen - lp_ref_chosen)
                    rejected_logratio = self.beta*(lp_cur_rejected - lp_ref_rejected)

                    h = chosen_logratio - rejected_logratio
                    return (1 - h).clamp(max=0)

        multi_pairwise_loss = torch.vmap(pairwise_loss, in_dims=(None, 0, None, 0)) # vmap over multiple rejections
        
        def listwise_loss(lp_cur_chosen, lp_ref_chosen, rank_chosen, lps_cur, lps_ref, ranks):
            valid_comps = ranks > rank_chosen
            comps = multi_pairwise_loss(lp_cur_chosen, lps_cur, lp_ref_chosen, lps_ref) * valid_comps
            return comps.sum() / (valid_comps.sum() + (valid_comps.sum() == 0))

        multi_listwise_losss = torch.vmap(listwise_loss, in_dims=(0, 0, 0, None, None, None))
        def batchwise_loss(lps_cur, lps_ref, ranks):
            valid_comps = (ranks != -100) * (ranks != torch.max(ranks))
            comps = multi_listwise_losss(lps_cur, lps_ref, ranks, lps_cur, lps_ref, ranks) * valid_comps
            return comps.sum() / (valid_comps.sum() + (valid_comps.sum() == 0))

        self.batched_loss = torch.vmap(batchwise_loss, in_dims=(0, 0, 0))

    def training_step(self, batch, batch_idx):
        ranks = batch['ranks']

        try:
            lp_cur, _ = self.get_logits(self.model, batch)
            with torch.no_grad():
                lp_ref = self.get_logits(self.ref_model, batch)
        except AssertionError as e:
            self.consecutive_nans += 1
            self.nan_counter += 1
            assert self.consecutive_nans <= self.skip_nans
            warnings.warn(f"NaNs or infs detected ({self.nan_counter} in training so far). Skipping batch")
            return None

        self.consecutive_nans = 0
        loss = self.batched_loss(lp_cur, lp_ref, ranks)

        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            logits, _ = self.get_logits(self.model, batch, normalize_length=False)
        except AssertionError as e:
            warnings.warn(f"NaNs or infs detected. Skipping batch")
            return None

        idxs = torch.argmax(logits)

        return (idxs == 0).float().mean()
