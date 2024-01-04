from .datamodule import DataModule
from transformers import PreTrainedTokenizer
import torch.multiprocessing as mp
from typing import Dict
from collections.abc import Sequence

NUM_PROC = mp.cpu_count() - 1

class Preprocessors:
    @staticmethod
    def simple_text(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_name: str ='text',
        num_proc: int = NUM_PROC,
        tokenizer_args: Sequence = [],
        tokenizer_kwargs: Dict = {},
        *args,
        **kwargs
    ):
        ds_dict = {}
        for k, v in datamodule.ds_dict.items():
            if v is not None:
                ds_dict[k] = v.map(
                    lambda x: tokenizer(x[column_name], *tokenizer_args, **tokenizer_kwargs),
                    remove_columns=v.column_names,
                    num_proc = num_proc,
                    *args, **kwargs
                )
            else:
                ds_dict[k] = None

        return DataModule.from_existing(datamodule, ds_dict=ds_dict)
