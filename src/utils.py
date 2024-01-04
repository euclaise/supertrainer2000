import torch
from typing import Callable, Type, TypeVar

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
