from __future__ import annotations
import lightning as L
import datasets
from typing import Optional, Callable, Union, Dict
from typing_extensions import TypedDict
from torch.utils.data import DataLoader, IterableDataset
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

class _DatasetSplitSizes(TypedDict):
    train: float
    test: float
    validation: float


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

        self.train_dataloader = self.create_dataloader('train')

        if self.ds_dict['validation'] != None:
            self.val_dataloader = self.create_dataloader('validation')

        return self

    def create_dataloader(self, split):
        return lambda: DataLoader(
            self.ds_dict[split],
            batch_size=self.batch_size,
            shuffle=split == 'train' and not self.streaming,
            collate_fn=self.collate_fn,
            num_workers=NUM_PROC if not self.streaming or self.ds_dict['train'].n_shards >= NUM_PROC else 1
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
            ds = datasets.load_from_disk(dataset, *args, **kwargs)
            streaming = False
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

        return cls(ds_dict, seed, streaming)

    @classmethod
    def merge(cls, to_merge):
        ds_dict = {}
        if any([ds.streaming for ds in to_merge]):
            raise ValueError('Cannot concatenate streamed datamodules, use datamodule.interleave instead')
        
        for k in ['train', 'test', 'validation']:
            if to_merge[0].ds_dict[k] is not None:
                ds_dict[k] = datasets.concatenate_datasets([d.ds_dict[k] for d in to_merge]).shuffle(seed=to_merge[0].seed)
            else:
                ds_dict[k] = None

        return cls.from_existing(to_merge[0], ds_dict=ds_dict)

    @classmethod
    def interleave(cls, to_concat: List, probabilities: List[float]):
        ds_dict = {}
        
        for k in ['train', 'test', 'validation']:
            if to_concat[0].ds_dict[k] is not None:
                ds_dict[k] = datasets.interleave_datasets([d.ds_dict[k] for d in to_concat], probabilities=probabilities, seed=self.seed).shuffle(seed=self.seed)
            else:
                ds_dict[k] = None

        return cls.from_existing(to_concat[0], ds_dict=ds_dict)


    def do_all(self, fn: Callable):
        ds_dict = {'train': None, 'test': None, 'validation': None}
        for k, v in self.ds_dict.items():
            if v is not None:
                ds_dict[k] = fn(v)

        return self.from_existing(self, ds_dict=ds_dict)

    def map(self, map_fn: Callable):
        if isinstance(self.ds_dict['train'], datasets.IterableDataset):
            return self.do_all(lambda ds: ds.map(map_fn, remove_columns=ds.column_names))
        return self.do_all(lambda ds: ds.map(map_fn, num_proc=NUM_PROC, remove_columns=ds.column_names))


    def truncate_toks(self, max_len):
        return self.map(lambda ds: {
            'input_ids': ds['input_ids'][:max_len],
            'labels': ds['labels'][:max_len]
        })

    def filter(self, filter_fn: Callable):
        if isinstance(self.ds_dict['train'], datasets.IterableDataset):
            return self.do_all(lambda ds: ds.filter(filter_fn))
        return self.do_all(lambda ds: ds.filter(filter_fn, num_proc=NUM_PROC))

    def shuffle(self):
        if self.streaming:
            warnings.warn("Streamed datasets cannot be fully shuffled -- performing a pseudo-shuffle instead.")
        return self.do_all(lambda ds: ds.shuffle(seed=self.seed))

    def create_splits(self, splits: _DatasetSplitSizes):
        p_sum = splits['train'] + splits['test'] + splits['validation']
        assert p_sum <= 1 and p_sum > 0, "Invalid probabilities for datamodule splits"

        splits1 = self.ds_dict['train'].train_test_split(test_size = splits['validation'], train_size = splits['train'] + splits['test'], seed=self.seed)
        val = splits1['test']
        splits2 = splits1['train'].train_test_split(test_size = splits['test'], train_size = splits['train'], seed=self.seed)

        ds_dict = {
            'train': splits2['train'],
            'test': splits2['test'],
            'validation': val
        }

        return self.from_existing(self, ds_dict=ds_dict)

    def take(self, ns):
        ds_dict = {'train': None, 'test': None, 'validation': None}
        for k, v in ns.items():
            ds_dict[k] = self.ds_dict[k].select(range(v))

        return self.from_existing(self, ds_dict=ds_dict)
