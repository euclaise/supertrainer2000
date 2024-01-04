from __future__ import annotations
import lightning as L
import datasets
from typing import Optional
from typing_extensions import TypedDict

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
        seed: int
    ):
        self._in_init = True
        super().__init__()
        
        self.ds_dict = ds_dict
        self.cache_dir = cache_dir
        self.seed = seed
        self._in_init = False

    def __str__(self):
        return self.ds_dict.__str__()

    def __repr__(self):
        return self.ds_dict.__repr__()

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
        if name == "_in_init" or self._in_init:
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
