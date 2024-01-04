from datasets import Dataset
from transformers import PreTrainedTokenizer
import torch.multiprocessing as mp

NUM_PROC = mp.cpu_count() - 1

class Preprocessors:
    @staticmethod
    def simple_text(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        column_name: str ='text',
        num_proc: int = NUM_PROC,
        *args,
        **kwargs
    ):
        return dataset.map(lambda x: tokenizer(x[column_name]), remove_columns=dataset.column_names, num_proc = num_proc, *args, **kwargs)
