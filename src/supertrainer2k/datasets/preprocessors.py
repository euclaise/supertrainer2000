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
        truncate: bool = False,
        tokenizer_kwargs: Dict = {},
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False

        if max_length != None and truncate:
            tokenizer_kwargs['max_length'] = max_length
            tokenizer_kwargs['truncation'] = True

        def fn(row):
            toks = tokenizer.encode(row[column_name], **tokenizer_kwargs)
            return {
                'input_ids': toks,
                'labels': toks
            }

        datamodule = datamodule.map(fn)

        if not truncate and max_length != None:
            datamodule = datamodule.filter(lambda row: len(row['input_ids']) <= max_length)
    
        return datamodule
        
    @staticmethod
    def multi_choice_text(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_names: Sequence[str, str] = ('chosen', 'rejected'),
        max_length: Optional[int] = None,
        truncate: bool = False,
        tokenizer_kwargs: Dict = {}
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False
            
        if max_length != None and truncate:
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

        datamodule = datamodule.map(map_fn)
        
        if not truncate and max_length != None:
            datamodule = datamodule.filter(lambda row: all(len(x) <= max_length for x in row['input_ids']))

        return datamodule


    @staticmethod
    def multi_choice_masked(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_names: Sequence[str, str] = ('text', 'masked', 'chosen', 'rejected'),
        max_length: Optional[int] = None,
        truncate: bool = False,
        external_ce_labels: Optional[str] = None,
        tokenizer_kwargs: Dict = {}
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False
            
        if max_length != None and truncate:
            tokenizer_kwargs['max_length'] = max_length
            tokenizer_kwargs['truncation'] = True


        def map_fn(row):
            def extract(resp):
                input_ids = []
                labels = []
                for segment in resp:
                    toks = tokenizer.encode(segment[column_names[0]], **tokenizer_kwargs)
                    input_ids += toks
                    labels += [-100]*len(toks) if segment[column_names[1]] else toks
                return input_ids, labels

            correct = extract(row[column_names[2]])
            incorrect = [extract(x) for x in row[column_names[3]]]
            if external_ce_labels is not None:
                gold = extract(row[external_ce_labels])
                return {
                    'ce_ids': gold[0],
                    'ce_labels': gold[1],
                    'input_ids': [correct[0]] + [i[0] for i in incorrect],
                    'labels': [correct[1]] + [i[1] for i in incorrect],
                    'ranks': [0] + [1]*len(incorrect),
                }
                
            return {
                'input_ids': [correct[0]] + [i[0] for i in incorrect],
                'labels': [correct[1]] + [i[1] for i in incorrect],
                'ranks': [0] + [1]*len(incorrect),
            }

        datamodule = datamodule.map(map_fn)
        
        if not truncate and max_length != None:
            datamodule = datamodule.filter(lambda row: all(len(x) <= max_length for x in row['input_ids']))

        return datamodule

        
    @staticmethod
    def ranked_masked(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_names: Sequence[str, str] = ('completions', 'messages', 'text', 'masked', 'rank'),
        max_length: Optional[int] = None,
        truncate: bool = False,
        tokenizer_kwargs: Dict = {}
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False
            
        if max_length != None and truncate:
            tokenizer_kwargs['max_length'] = max_length
            tokenizer_kwargs['truncation'] = True

        def map_fn(row):
            def extract(resp):
                input_ids = []
                labels = []
                rank = resp[column_names[4]]
                
                for segment in resp[column_names[1]]:
                    toks = tokenizer.encode(segment[column_names[2]], **tokenizer_kwargs)
                    input_ids += toks
                    labels += [-100]*len(toks) if segment[column_names[3]] else toks
                return input_ids, labels, rank

            completions = [extract(x) for x in row[column_names[0]]]

            return {
                'input_ids': [c[0] for c in completions],
                'labels': [c[1] for c in completions],
                'ranks': [c[2] for c in completions]
            }

        datamodule = datamodule.map(map_fn)
        
        if not truncate and max_length != None:
            datamodule = datamodule.filter(lambda row: all(len(x) <= max_length for x in row['input_ids']))

        return datamodule

        
    @staticmethod
    def ranked_text(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_name: str = 'completions',
        completions_keys: Sequence[str, str] = ('text', 'rank'),
        max_length: Optional[int] = None,
        truncate: bool = False,
        tokenizer_kwargs: Dict = {}
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False
            
        if max_length != None and truncate:
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

        datamodule = datamodule.map(map_fn)
        
        if not truncate and max_length != None:
            datamodule = datamodule.filter(lambda row: all(len(x) <= max_length for x in row['input_ids']))

        return datamodule

    @staticmethod
    def multi_instruction_response(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_name: str = 'messages',
        role_names: Sequence[str, str] = ('user', 'assistant', 'system'),
        msg_keys: Sequence[str, str] = ('content', 'role'),
        max_length: Optional[int] = None,
        truncate: bool = False,
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
                if msg[msg_keys[1]] ==  role_names[0] or msg[msg_keys[1]] == role_names[2]:
                    labels += [-100]*len(toks)
                elif msg[msg_keys[1]] == role_names[1]:
                    labels += toks
                else:
                    raise ValueError(f"Unknown role name, {msg[msg_keys[1]]}")

            if truncate and max_length != None:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
    
            return {
                'input_ids': input_ids,
                'labels': labels,
            }
            
        datamodule = datamodule.map(map_fn)
            
        if not truncate and max_length != None:
            datamodule = datamodule.filter(lambda row: len(row['input_ids']) <= max_length)
            
        return datamodule


    @staticmethod
    def instruction_response(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_names: Sequence[str, str] = ('instruction', 'response'),
        max_length: Optional[int] = None,
        truncate: bool = False,
        tokenizer_kwargs: Dict = {},
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False

        def map_fn(row):
            instruction = tokenizer.encode(row[column_names[0]], **tokenizer_kwargs)
            response = tokenizer.encode(row[column_names[1]], **tokenizer_kwargs)

            input_ids = instruction + response
            labels = [-100]*len(instruction) + response

            if truncate and max_length != None:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
    
            return {
                'input_ids': input_ids,
                'labels': labels,
            }
            
        datamodule = datamodule.map(map_fn)

        if not truncate and max_length != None:
            datamodule = datamodule.filter(lambda row: len(row['input_ids']) <= max_length)

        return datamodule


    @staticmethod
    def stepbystep_instruction_response(
        datamodule: DataModule,
        tokenizer: PreTrainedTokenizer,
        column_names: Sequence[str, str] = ('instruction_rationale', 'instruction_answer', 'rationale', 'answer'),
        max_length: Optional[int] = None,
        truncate: bool = False,
        tokenizer_kwargs: Dict = {},
    ):
        if 'add_special_tokens' not in tokenizer_kwargs:
            tokenizer_kwargs['add_special_tokens'] = False

        def map_fn(row):
            instruction_rationale = tokenizer.encode(row[column_names[0]], **tokenizer_kwargs)
            instruction_answer = tokenizer.encode(row[column_names[1]], **tokenizer_kwargs)

            rationale = tokenizer.encode(row[column_names[2]], **tokenizer_kwargs)
            answer = tokenizer.encode(row[column_names[3]], **tokenizer_kwargs)
            
            rationale_ids = instruction_rationale + rationale
            answer_ids = instruction_answer + answer
            rationale_labels = [-100]*len(instruction_rationale) + rationale
            answer_labels = [-100]*len(instruction_answer) + answer

            if truncate and max_length != None:
                rationale_ids = rationale_ids[:max_length]
                rationale_labels = rationale_labels[:max_length]
                answer_ids = answer_ids[:max_length]
                answer_labels = answer_labels[:max_length]
    
            return {
                'rationale_ids': rationale_ids,
                'answer_ids': answer_ids,
                'rationale_labels': rationale_labels,
                'answer_labels': answer_labels,
            }
            
        datamodule = datamodule.map(map_fn)

        if not truncate and max_length != None:
            datamodule = datamodule.filter(lambda row: len(row['rationale_ids']) <= max_length and len(row['answer_ids']) <= max_length)

        return datamodule
