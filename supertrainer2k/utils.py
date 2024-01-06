import torch
from typing import Callable, Type, TypeVar, Dict, Any, Optional, Union, List
from lightning.pytorch.strategies import Strategy

TorchModule = TypeVar('TorchModule', bound=torch.nn.Module)

def patch_model(
    model: torch.nn.Module,
    to_replace: List[Type[TorchModule]],
    patch_fn: Callable[[TorchModule], TorchModule],
    name_constraints: Callable[[str], bool] = lambda x: True
):
    """Replaces all `to_replace` submodules which meet `name_constraints` with `replace_fn(submodule)`"""
    for name, submodule in model.named_children():
        for r in to_replace:
            if isinstance(submodule, r) and name_constraints(name):
                setattr(model, name, patch_fn(submodule))
                continue
            patch_model(submodule, to_replace, patch_fn, name_constraints)


class DummyStrategy(Strategy): # Dummy strategy to keep HF's device placement                                                                                                                                                                         
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_rank = 0                                                                                                                                                                                      

    def model_to_device(self) -> None:
        pass

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0) -> Any:
        device = self.model.device if self.model else (device or self.root_device)
        return move_data_to_device(batch, device)

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        pass

    def barrier(self, name: Optional[str] = None) -> None:
        pass

    def broadcast(self, obj, src: int = 0):
        return obj

    @property
    def is_global_zero(self) -> bool:
        return True

    def reduce(
        self,
        tensor: Union[torch.Tensor, Any],
        group: Optional[Any] = None,
        reduce_op = "mean",
    ) -> Union[torch.Tensor, Any]:
        return tensor

    @property
    def root_device(self) -> torch.device:
        return torch.device('cuda:0')
