from __future__ import annotations
import lightning as L
import datasets
from typing import Optional, Callable, Union
from typing_extensions import TypedDict
from torch.utils.data import DataLoader
from ..config import NUM_PROC
import warnings

class _DatasetSplits(TypedDict):
    train: str
    test: Optional[str]
    validation: Optional[str]


class _DatasetDict(TypedDict):
    train: Union[datasets.Dataset, datasets.IterableDataset]
    test: Optional[Union[datasets.Dataset, datasets.IterableDataset]]
    validation: Optional[Union[datasets.Dataset, datasets.IterableDataset]]

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        ds_dict: _DatasetDict,
        seed: int,
        streaming: bool
    ):
        self._immutable = False
        super().__init__()
        
        self.ds_dict = ds_dict
        self.seed = seed
        self.streaming = streaming

        self._immutable = True

    def __str__(self):
        return self.ds_dict.__str__()

    def __repr__(self):
        return self.ds_dict.__repr__()

    def init(self, batch_size: int, collate_fn: Callable):
        self._immutable = False
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        return self

    def train_dataloader(self):
        return DataLoader(
            self.ds_dict['train'],
            batch_size=self.batch_size,
            shuffle=not self.streaming,
            collate_fn=self.collate_fn,
            num_workers=NUM_PROC
        )

    @classmethod
    def from_existing(
        cls,
        existing: DataModule,
        ds_dict: Optional[DatasetDict] = None,
        seed: Optional[int] = None,
        streaming: Optional[bool] = None
    ):
        ds_dict = ds_dict if ds_dict is not None else existing.ds_dict
        seed = seed if seed is not None else existing.seed
        streaming = streaming if streaming is not None else existing.streaming

        return cls(ds_dict, seed, streaming)

    def __setattr__(self, name, value):
        if name == "_immutable" or not self._immutable:
            super().__setattr__(name, value)
        else:
            raise AttributeError("supertrainer2000 DataModules are immutable. Attempted to modify immutable data.")
    
    @classmethod
    def load_hf(
        cls,
        dataset: str,
        splits: DatasetSplits = {'train': 'train', 'test': None, 'validation': None},
        seed: int = 42,
        streaming: bool = False,
        *args,
        **kwargs
    ):
        if dataset.startswith(('/', './')):
            ds = datasets.load_from_disk(dataset, streaming=streaming, *args, **kwargs)
        else:
            ds = datasets.load_dataset(dataset, streaming=streaming, *args, **kwargs)

        if isinstance(ds, datasets.DatasetDict):
            ds_dict = {
                'train': ds[splits['train']],
                'test': None if splits['test'] is None else ds[splits['test']],
                'validation': None if splits['validation'] is None else ds[splits['validation']]
            }
        else:
            ds_dict = {
                'train': ds,
                'test': None,
                'validation': None
            }
        seed = seed
        streaming = streaming

        return cls(ds_dict, seed, streaming)

    @classmethod
    def concatenate(cls, to_concat):
        ds_dict = {}
        
        for k in ['train', 'test', 'validation']:
            if to_concat[0].ds_dict[k] is not Null:
                ds_dict[k] = datasets.concatenate_datasets([d.ds_dict[k] for d in to_concat]).shuffle(seed=self.seed)
            else:
                ds_dict[k] = None

        return cls.from_existing(to_concat[0], ds_dict=ds_dict)


    def do_all(self, fn: Callable):
        ds_dict = {}
        for k, v in self.ds_dict.items():
            if v is not None:
                ds_dict[k] = fn(v)

        return self.from_existing(self, ds_dict=ds_dict)

    def map(self, map_fn: Callable):
        if isinstance(self.ds_dict['train'], datasets.IterableDataset):
            return self.do_all(lambda ds: ds.map(map_fn, remove_columns=ds.column_names))
        return self.do_all(lambda ds: ds.map(map_fn, num_proc=NUM_PROC, remove_columns=ds.column_names))


    def truncate_toks(self, max_len):
        return self.map(lambda ds: {'toks': ds['toks'][:max_len]})

    def filter(self, filter_fn: Callable):
        if isinstance(self.ds_dict['train'], datasets.IterableDataset):
            return self.do_all(lambda ds: ds.filter(filter_fn))
        return self.do_all(lambda ds: ds.filter(filter_fn, num_proc=NUM_PROC))

    def shuffle(self):
        if self.streaming:
            warnings.warn("Streamed datasets cannot be fully shuffled -- performing a pseudo-shuffle instead.")
        return self.do_all(lambda ds: ds.shuffle(seed=self.seed))
