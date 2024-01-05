import torch
from typing import Callable, Type, TypeVar, Dict
from transformers import get_scheduler, PreTrainedTokenizer
import torch
from collections.abc import Sequence

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

def get_hf_scheduler(
    name: str,
    num_warmup_steps = 0,
    **kwargs
):
    def scheduler_fn(optimizer, total_steps):
        return get_scheduler(
            name,
            optimizer,
            num_warmup_steps = num_warmup_steps,
            num_training_steps = total_steps,
            scheduler_specific_kwargs = kwargs
        )

    return scheduler_fn


class DataCollatorForCausal:
    tokenizer: PreTrainedTokenizer

    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        toks = [d['toks'] for d in instances]

        max_seq_len = max(len(x) for x in toks)

        input_ids = [torch.tensor([self.pad_id]*(max_seq_len - len(x)) + x, dtype=torch.long) for x in toks]
        input_ids = torch.stack(input_ids)
        
        labels = [torch.tensor([-100]*(max_seq_len - len(x)) + x, dtype=torch.long) for x in toks]
        labels = torch.stack(labels)

        attention_mask = [torch.tensor([False]*(max_seq_len - len(x)) + [True]*len(x), dtype=torch.bool) for x in toks]
        attention_mask = torch.stack(attention_mask)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )