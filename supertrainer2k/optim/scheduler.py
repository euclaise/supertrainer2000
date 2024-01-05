from torch.optim import Optimizer
from transformers import get_scheduler

def get_hf_scheduler(
    name: str,
    num_warmup_steps: int = 0,
    **kwargs
):
    def scheduler_fn(optimizer: Optimizer, total_steps: int):
        return get_scheduler(
            name,
            optimizer,
            num_warmup_steps = num_warmup_steps,
            num_training_steps = total_steps,
            scheduler_specific_kwargs = kwargs
        )

    return scheduler_fn
