import lightning as L
import torch
import transformers
from torch.optim import Optimizer,  lr_scheduler
from typing import Optional, Dict, Literal, Callable
from ..datasets import DataModule
from ..optim.dummy import _DummyOptimizer
from torch.distributed.optim.apply_optimizer_in_backward import _apply_optimizer_in_backward

class Wrapper(L.LightningModule):
    def __init__(self,
        model: transformers.PreTrainedModel,
        optimizer: Optimizer,
        scheduler: Callable,
        lr: float,
        tokenizer,
        adalite_backward: bool = False,
        scheduler_config: Optional[Dict] = None,
        optimizer_params: Optional[Dict] = {},
        clip_outputs: Optional[float] = None,
        skip_nans: int = 10,
    ):
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer
        self.scheduler = scheduler
        self.optimizer_params = optimizer_params
        self.lr = lr
        self.adalite_backward = adalite_backward
        self.clip_outputs = clip_outputs
        self.nan_counter = 0
        self.skip_nans = skip_nans
        self.consecutive_nans = 0

        if scheduler_config is not None:
            self.scheduler_config = scheduler_config
        else:
            self.scheduler_config = {
                'interval': 'step',
                'frequency': 1
            }
        self.tokenizer = tokenizer
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_logits(self, model, batch, normalize_length = True):
        orig_shape = batch['input_ids'].shape
        seq_len = orig_shape[-1]
        
        flat_input_ids = batch['input_ids'].view(-1, seq_len)[:, :-1]
        flat_attn_mask = batch['attention_mask'].view(-1, seq_len)[:, :-1]
        flat_labels = batch['labels'].view(-1, seq_len)[:, 1:]

        flat_attn_mask_o = torch.where((flat_attn_mask.sum(dim=-1) == 0).unsqueeze(-1), 1, flat_attn_mask)

        logits = model(input_ids=flat_input_ids, attention_mask=flat_attn_mask_o).logits
        if self.clip_outputs:
            logits = logits.clip(-self.clip_outputs, self.clip_outputs)

        logits = logits.log_softmax(dim=-1)

        assert not torch.isnan(logits).any() and not torch.isinf(logits).any()

        mask = (flat_labels != -100) * (flat_attn_mask != 0)
        logits = torch.gather(logits, -1, torch.where(mask, flat_labels, 0).unsqueeze(-1)).squeeze(-1) * mask
        new_shape = orig_shape[:-1] + (seq_len - 1,)
        logits = logits.view(new_shape).sum(dim=-1)

        n = mask.view(new_shape).sum(dim=-1)

        if not normalize_length:
            return logits, (n != 0)

        return torch.where(n != 0, logits / n, 0), (n != 0)
       
    def configure_optimizers(self):
        if self.adalite_backward:
            if self.trainer.accumulate_grad_batches > 1:
                raise ValueError("Gradient accumulation is incompatible with Adalite-style backwarsds passes. Please pass adalite_backward=False to your Wrapper, or do not use gradient accumulation.")

            self.optimizer_params['lr'] = self.lr
            _apply_optimizer_in_backward(
                self.optimizer_cls,
                self.model.parameters(),
                optimizer_kwargs = self.optimizer_params
            )

            self.optimizer = _DummyOptimizer(self.model.parameters())
        else:  
            self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr, **self.optimizer_params)

        scheduler = self.scheduler_config
        total_steps = self.trainer.estimated_stepping_batches if scheduler['interval'] == 'step' else self.trainer.max_epochs

        scheduler['scheduler'] = self.scheduler(self.optimizer, total_steps=total_steps)


        return [self.optimizer], [scheduler]

    def save(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        self.model.push_to_hub(*args, **kwargs)
