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

        def map_fn(row):
            chosen = [tokenizer.encode(row[column_names[0]], **tokenizer_kwargs)]
            rejected = [tokenizer.encode(c, **tokenizer_kwargs) for c in row[column_names[1]]]

            return {
                'input_ids': chosen + rejected,
                'labels': chosen + rejected,
                'ranks': [0] + [1]*len(rejected),
            }

        return datamodule.map(map_fn)

        
    @staticmethod
    def ranked_text(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_name: str = 'completions',
        completions_keys: Sequence[str, str] = ('text', 'rank'),
        max_length: Optional[int] = None,
        tokenizer_kwargs: Dict = {}
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False
            
        if max_length != None:
            tokenizer_kwargs['max_length'] = max_length
            tokenizer_kwargs['truncation'] = True

        def map_fn(row):
            input_ids = [tokenizer.encode(seq[completions_keys[0]], **tokenizer_kwargs) for seq in row[column_name]]
            ranks = [seq['rank'] for seq in row[column_name]]
            return {
                'input_ids': input_ids,
                'labels': input_ids,
                'ranks': ranks,
            }

        return datamodule.map(map_fn)

    @staticmethod
    def multi_instruction_response(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_name: str = 'messages',
        role_names: Sequence[str, str] = ('instruction', 'response'),
        msg_keys: Sequence[str, str] = ('text', 'role'),
        max_length: Optional[int] = None,
        tokenizer_kwargs: Dict = {},
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False

        def map_fn(row):
            input_ids = []
            labels = []

            for msg in row[column_name]:
                toks = tokenizer.encode(msg[msg_keys[0]], **tokenizer_kwargs)
                input_ids += toks
                if msg[msg_keys[1]] ==  role_names[0]:
                    labels += [-100]*len(toks)
                elif msg[msg_keys[1]] == role_names[1]:
                    labels += toks
                else:
                    raise ValueError(f"Unknown role name, {msg[msg_keys[1]]}")

            return {
                'input_ids': input_ids[:max_length],
                'labels': labels[:max_length],
            }
            
        return datamodule.map(map_fn)


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

        def map_fn(row):
            instruction = tokenizer.encode(x[column_names[0]], **tokenizer_kwargs)
            response = tokenizer.encode(x[column_names[1]], **tokenizer_kwargs)

            return {
                'input_ids': (instruction + response)[:max_length],
                'labels': ([-100]*len(instruction) + response)[:max_length]
            }
            
        return datamodule.map(map_fn)
