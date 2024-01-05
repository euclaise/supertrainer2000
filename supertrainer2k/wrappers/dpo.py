import lightning as L
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from .wrapper import _Wrapper
import torch

class DPOWrapper(_Wrapper):
    def __init__(self, ref_model: torch.nn.Module, beta: float = 0.1, eps: float = 1.0, *args, **kwargs):
        self.beta = beta
        self.eps = eps
        self.ref_model = copy.deepcopy(ref_model)
        self.ref_model.eval()
        super().__init__(*args, **kwargs)

    def forward(self, **inputs):
        return self.model(**inputs)

    def dpo_loss(self, logprobs, ref_logprobs, mask):
    
        pol_logratios = logprobs[:, 0].unsqueeze(-1) - logprobs[:, 1:]
        ref_logratios = reff_logprobs[:, 0].unsqueeze(-1) - ref_logprobs[:, 1:]

        losses = -F.logsigmoid(self.beta * h) * (1 - self.eps) - F.logsigmoid(-self.beta * h)*self.eps

        return losses * mask[:, 1:]

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
        lp_cur = self.get_logits(self.model, batch)
        with torch.no_grad():
            lp_ref = self.get_logits(self.ref_model, batch)

        mask_sum = batch['mask'].sum()

        if mask_sum == 0:
            raise ValueError('Batch contains no unmasked sequences')
        
        losses = self.dpo_loss(lp_cur, batch['mask']).sum() / mask_sum

        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):    
        logits = self.get_logits(batch, normalize_length=False)

        # Logits is of shape [bsz, comps]
        idxs = torch.argmax(logits)

        return (idxs == 0).float().mean()
