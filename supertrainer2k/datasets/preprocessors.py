from .datamodule import DataModule
from transformers import PreTrainedTokenizer
from typing import Dict, Optional
from collections.abc import Sequence
from ..config import NUM_PROC

class Preprocessors:
    @staticmethod
    def simple_text(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_name: str ='text',
        max_length: Optional[int] = None,
        tokenizer_kwargs: Dict = {},
    ):
        if max_length != None:
            tokenizer_kwargs['max_length'] = max_length
            tokenizer_kwargs['truncation'] = True

        return datamodule.map(
            lambda x: {'toks': tokenizer.encode(x[column_name], **tokenizer_kwargs)},
        )
        
    @staticmethod
    def multi_choice_text(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_names: Sequence[str, str] = ('chosen', 'rejected'),
        max_length: Optional[int] = None,
        tokenizer_kwargs: Dict = {}
    ):
        if max_length != None:
            tokenizer_kwargs['max_length'] = max_length
            tokenizer_kwargs['truncation'] = True
        return datamodule.map(
            lambda x: {
                'toks_chosen': tokenizer.encode(x[column_names[0]], **tokenizer_kwargs),
                'toks_rejected': tokenizer(x[column_names[1]], **tokenizer_kwargs).input_ids
            }
        )
