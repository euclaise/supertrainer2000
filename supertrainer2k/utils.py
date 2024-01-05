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

class DummyStrategy(Strategy): # Dummy strategy to keep HF's device placement                                                                                                                                                                         
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_rank = 0                                                                                                                                                                                      

    def model_to_device(self) -> None:
        pass

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0) -> Any:
        device = self.model.device if self.model else (device or self.root_device)
        return move_data_to_device(batch, device)

    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
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
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op = "mean",
    ) -> Union[Tensor, Any]:
        return tensor

    @property
    def root_device(self) -> torch.device:
        return torch.device('cuda:0')
