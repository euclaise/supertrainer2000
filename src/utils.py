import torch
from typing import Callable, Type, TypeVar
from transformers import get_scheduler

TorchModule = TypeVar('TorchModule', bound=torch.nn.Module)

def find_and_replace(
    module: torch.nn.Module,
    to_replace: Type[TorchModule],
    replace_fn: Callable[[TorchModule], TorchModule],
    name_constriants: Callable[[str], bool] = lambda x: True
):
    """Replaces all `to_replace` submodules which meet `name_constraints` with `replace_fn(submodule)`"""
    for name, submodule in module.named_children():
        if isinstance(submodule, to_replace) and name_constraints(name):
            setattr(module, name, replace_fn(submodule))
        else:
            find_and_replace(submodule, replacement)

def get_hf_scheduler(
    name: str,
    num_warmup_steps = 0,
    **kwargs
):
    def scheduler_fn(optimizer, total_steps):
        scheduler = get_scheduler(
            scheduler_name,
            optimizer,
            num_warmup_steps = num_warmup_steps,
            num_training_steps = total_steps,
            scheduler_specific_kwargs = kwargs
        )

    return scheduler_fn
