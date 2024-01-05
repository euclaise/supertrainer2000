from .datamodule import DataModule
from transformers import PreTrainedTokenizer
from typing import Dict, Optional
from collections.abc import Sequence
from ..config import NUM_PROC


    
class Preprocessor:
    @staticmethod
    def simple_text(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_name: str ='text',
        max_length: Optional[int] = None,
        tokenizer_kwargs: Dict = {},
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False
            
        if max_length != None:
            tokenizer_kwargs['max_length'] = max_length
            tokenizer_kwargs['truncation'] = True

        def fn(text, **kwargs):
            toks = tokenizer.encode(text, **tokenizer_kwargs)
            return {
                'input_ids': toks,
                'labels': toks
            }
    
        return datamodule.map(
            lambda x: fn(x[column_name]),
        )
        
    @staticmethod
    def multi_choice_text(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_names: Sequence[str, str] = ('chosen', 'rejected'),
        max_length: Optional[int] = None,
        tokenizer_kwargs: Dict = {}
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False
            
        if max_length != None:
            tokenizer_kwargs['max_length'] = max_length
            tokenizer_kwargs['truncation'] = True

        def fn(text, **kwargs):
            toks = tokenizer.encode(text, **tokenizer_kwargs)
            return {
                'input_ids': toks,
                'labels': toks
            }

        return datamodule.map(
            lambda x: {
                'chosen': fn(x[column_names[0]]),
                'rejected': [fn(c) for c in x[column_names[1]]]
            }
        )

    @staticmethod
    def instruction_response(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_names: Sequence[str, str] = ('instruction', 'response'),
        max_length: Optional[int] = None,
        tokenizer_kwargs: Dict = {},
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False

        datamodule = datamodule.map(
            lambda x: {
                'instruction': tokenizer.encode(x[column_names[0]], **tokenizer_kwargs),
                'response': tokenizer.encode(x[column_names[1]], **tokenizer_kwargs)
            }
        )

        return datamodule.map(
            lambda x: {
                'input_ids': (x['instruction'] + x['response'])[:max_length],
                'labels': ([-100]*len(x['instruction']) + x['response'])[:max_length]
            }
        )
