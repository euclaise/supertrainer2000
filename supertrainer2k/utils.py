import torch
from typing import Callable, Type, TypeVar, Dict
import torch

TorchModule = TypeVar('TorchModule', bound=torch.nn.Module)

def patch_model(
    model: torch.nn.Module,
    to_replace: Type[TorchModule],
    patch_fn: Callable[[TorchModule], TorchModule],
    name_constriants: Callable[[str], bool] = lambda x: True
):
    """Replaces all `to_replace` submodules which meet `name_constraints` with `replace_fn(submodule)`"""
    for name, submodule in module.named_children():
        if isinstance(submodule, to_replace) and name_constraints(name):
            setattr(module, name, patch_fn(submodule))
        else:
            patch_model(submodule, replacement)
