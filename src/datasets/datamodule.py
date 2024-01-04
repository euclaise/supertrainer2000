from __future__ import annotations
import lightning as L
import datasets
from typing import Optional
from typing_extensions import TypedDict
from torch.utils.data import DataLoader
from ..config import NUM_PROC

class _DatasetSplits(TypedDict):
    train: str
    test: Optional[str]
    validation: Optional[str]


class _DatasetDict(TypedDict):
    train: datasets.Dataset
    test: Optional[datasets.Dataset]
    validation: Optional[datasets.Dataset]

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        ds_dict: _DatasetDict,
        cache_dir: str,
        seed: int,
    ):
        self._immutable = False
        super().__init__()
        
        self.ds_dict = ds_dict
        self.cache_dir = cache_dir
        self.seed = seed

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
        return DataLoader(self.ds_dict['train'], batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=NUM_PROC)

    @classmethod
    def from_existing(
        cls,
        existing: DataModule,
        ds_dict: Optional[DatasetDict] = None,
        cache_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        ds_dict = ds_dict if ds_dict is not None else existing.ds_dict
        cache_dir = cache_dir if cache_dir is not None else existing.cache_dir
        seed = seed if seed is not None else existing.seed

        return cls(ds_dict, cache_dir, seed)

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
        cache_dir: str = "./ds_cache",
        seed: int = 42,
        *args,
        **kwargs
    ):
        if dataset.startswith(('/', './')):
            ds = datasets.load_from_disk(dataset, *args, **kwargs)
        else:
            ds = datasets.load_dataset(dataset, *args, **kwargs)

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
        cache_dir = cache_dir
        seed = seed

        return cls(ds_dict, cache_dir, seed)

    @classmethod
    def concatenate(cls, to_concat):
        ds_dict = {}
        
        for k in ['train', 'test', 'validation']:
            if to_concat[0].ds_dict[k] is not Null:
                ds_dict[k] = datasets.concatenate_datasets([d.ds_dict[k] for d in to_concat]).shuffle(seed=self.seed)
            else:
                ds_dict[k] = None

        return cls.from_existing(to_concat[0], ds_dict=ds_dict)


    def truncate_len(self, max_len):
        ds_dict = {}
        for k, v in self.ds_dict.items():
            if v is not None:
                ds_dict[k] = v.map(lambda x: {'toks': x['toks'][:max_len]})

        return self.from_existing(self, ds_dict=ds_dict)
