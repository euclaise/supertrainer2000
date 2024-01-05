from .datamodule import DataModule
from transformers import PreTrainedTokenizer
from typing import Dict
from collections.abc import Sequence
from ..config import NUM_PROC

class Preprocessors:
    @staticmethod
    def simple_text(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_name: str ='text',
        tokenizer_args: Sequence = [],
        tokenizer_kwargs: Dict = {}
    ):
        return datamodule.map(
            lambda x: {'toks': tokenizer.encode(x[column_name], *tokenizer_args, **tokenizer_kwargs)},
        )
        
    @staticmethod
    def multi_choice_text(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_names: Sequence[str, str] = ('chosen', 'rejected'),
        tokenizer_args: Sequence = [],
        tokenizer_kwargs: Dict = {}
    ):
        return datamodule.map(
            lambda x: {
                'toks_chosen': tokenizer.encode(x[column_names[0]], *tokenizer_args, **tokenizer_kwargs),
                'toks_rejected': tokenizer(x[column_names[1]], *tokenizer_args, **tokenizer_kwargs).input_ids
            }
        )
