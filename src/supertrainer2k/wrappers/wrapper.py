import lightning as L
import torch
import torch.nn.functional as F
import transformers
from torch.optim import Optimizer,  lr_scheduler
from typing import Optional, Dict, Literal, Callable
from ..datasets import DataModule
from ..optim.dummy import _DummyOptimizer
from torch.distributed.optim.apply_optimizer_in_backward import _apply_optimizer_in_backward
from typing import NamedTuple


class ElasticResetArgs(NamedTuple):
    eta: float
    interval: int

class Wrapper(L.LightningModule):
    """
    The `Wrapper` class defines basic wrapper properties, and is subclassed by classes such as `SFTWrapper` which wrap around Huggingface transformers models.
    """
    def __init__(self,
        model: transformers.PreTrainedModel,
        optimizer: Optimizer,
        scheduler: Callable,
        lr: float,
        adalite_backward: bool = False,
        scheduler_config: Optional[Dict] = None,
        optimizer_args: Optional[Dict] = {},
        clip_outputs: Optional[float] = None,
        skip_nans: int = 3,
        freeze_embeds: bool = False,
        elastic_reset_args: Optional[ElasticResetArgs] = None
    ):
        """
        Parameters:
            model (transformers.PreTrainedModel): The transformer model to be trained.
            optimizer_cls (Optimizer): The class of the optimizer to use for training.
            scheduler (Callable): The learning rate scheduler function.
            lr (float): The learning rate.
            adalite_backward (bool): Flag to use Adalite-style backward passes, where the parameter updates are fused with the gradient computations. Default is False.
            scheduler_config (Optional[Dict]): Configuration for the scheduler. Default is None.
            optimizer_args (Optional[Dict]): Additional arguments for the optimizer. Default is an empty dictionary.
            clip_outputs (Optional[float]): Maximum value to which outputs are clipped. This clips the model outputs PRIOR to softmax, which can be helpful in the case of NaN/inf values caused by exploding logits. Default is None.
            skip_nans (int): Number of consecutive NaN/inf values to ignore during training before error. Default is 3.
            freeze_embeds (bool): Freeze input and output embeddings. This is recommended for finetuning unless you are adding custom tokens. Default is False.
        """
        super().__init__()
        self.model = model
        if freeze_embeds:
            for p in self.model.get_input_embeddings().parameters():
                p.requires_grad = False
            for p in self.model.get_output_embeddings().parameters():
                p.requires_grad = False
        
        self.optimizer_cls = optimizer
        self.scheduler = scheduler
        self.optimizer_args = optimizer_args
        self.lr = lr
        self.adalite_backward = adalite_backward
        self.clip_outputs = clip_outputs
        self.nan_counter = 0
        self.skip_nans = skip_nans
        self.consecutive_nans = 0
        self.elastic_reset_args = elastic_reset_args
        if elastic_reset_args:
            self.ema_model = copy.deepcopy(model)
            self.ema_step_count = 0

        if scheduler_config is not None:
            self.scheduler_config = scheduler_config
        else:
            self.scheduler_config = {
                'interval': 'step',
                'frequency': 1
            }
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def ema_update(self):
        eta = self.elastic_reset_args.eta
        with torch.no_grad():
            for (ema_p, p) in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data = (1 - eta)*p.data + eta*ema_p.data

    def ema_reset(self):
        if self.elastic_reset_args:
            self.model.load_state_dict(self.ema_model.state_dict())
            self.ema_model = copy.deepcopy(self.model)
            self.ema_step_counter = 0

    def get_logits(self, model, batch):
        orig_shape = batch['input_ids'].shape
        seq_len = orig_shape[-1]
        
        flat_input_ids = batch['input_ids'].view(-1, seq_len)[:, :-1].contiguous()
        if 'attention_mask' in batch:
            flat_attn_mask = batch['attention_mask'].view(-1, seq_len)[:, :-1].contiguous()
        else:
            flat_attn_mask = torch.ones_like(flat_input_ids).bool()
        flat_labels = batch['labels'].view(-1, seq_len)[:, 1:].contiguous()

        flat_attn_mask_o = torch.where((flat_attn_mask.sum(dim=-1) == 0).unsqueeze(-1), 1, flat_attn_mask)

        logits = model(input_ids=flat_input_ids, attention_mask=flat_attn_mask_o).logits
        
        if self.clip_outputs:
            logits = logits.clip(-self.clip_outputs, self.clip_outputs)

        logits = logits.log_softmax(dim=-1)
        mask = (flat_labels != -100)
        flat_labels[~mask] = 0

        logits = logits.gather(-1, flat_labels.unsqueeze(-1))


        logits[~mask] = 0
        new_shape = orig_shape[:-1] + (seq_len - 1,)
        mask = mask.view(new_shape)
        logits = logits.view(new_shape)

        return logits, mask

    def ema_step(self):
        if self.elastic_reset_args:
            self.ema_update()
            self.ema_step_counter += 1
            if self.ema_step_counter == self.elastic_reset_args.interval:
                self.ema_reset()
            
        
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.adalite_backward:
            self.optimizer_args['lr'] = self.lr
            _apply_optimizer_in_backward(
                self.optimizer_cls,
                params,
                optimizer_kwargs = self.optimizer_args
            )

            self.optimizer = _DummyOptimizer(params, lr=self.lr)
        else:  
            self.optimizer = self.optimizer_cls(params, lr=self.lr, **self.optimizer_args)

        scheduler = self.scheduler_config
        total_steps = self.trainer.estimated_stepping_batches if scheduler['interval'] == 'step' else self.trainer.max_epochs

        scheduler['scheduler'] = self.scheduler(self.optimizer, total_steps=total_steps)


        return [self.optimizer], [scheduler]

    def save(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        self.model.push_to_hub(*args, **kwargs)
