from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler
from typing import List, Tuple, Callable

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

def get_chained_scheduler(schedulers: List[Tuple[Callable, int]]):
    """
    Create a chained scheduler function that activates different schedulers at specified intervals.

    Args:
        schedulers (List[Tuple[Callable, int]]): A list of tuples, each containing a scheduler function,
                                                 and the starting step for the use of that scheduler.

    Returns:
        Callable: A function that takes an optimizer and total steps and returns a LambdaLR scheduler.
    """
    def scheduler_fn(optimizer: Optimizer, total_steps: int):
        used_schedulers = {}
        def lr_lambda(current_step):
            active_scheduler = None
            for i, (scheduler, start_step) in enumerate(schedulers):
                if start_step is None or current_step >= start_step:
                    if i + 1 < len(schedulers) and schedulers[i + 1][1] is not None and  current_step >= schedulers[i + 1][1]:
                            continue
                    active_scheduler = scheduler

            if active_scheduler:
                if scheduler not in used_schedulers:
                    used_schedulers[active_scheduler] = active_scheduler(optimizer, total_steps).lr_lambdas[0]
                return used_schedulers[active_scheduler](current_step)

            return 0

        return LambdaLR(optimizer, lr_lambda)

    return scheduler_fn
