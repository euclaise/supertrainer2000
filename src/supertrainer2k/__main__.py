import tomllib
import sys
from .datasets import DataModule, Preprocessor, DataCollator
from .wrappers import SFTWrapper, PROWrapper
from .optim import Adalite, get_hf_scheduler

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from torch import optim

def load_config(path):
    with open(path, 'rb') as f:
        config = tomllib.load(f)
    return config

def load_model(config):
    if 'model' not in config:
        raise ValueError("Missing [model] section in the configuration")

    model_name = config['model']['name']
    dtype = getattr(torch, config['model'].get('dtype', 'bfloat16'))

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def load_dataset(config, tokenizer):
    if 'dataset' not in config:
        raise ValueError("Missing [dataset] section in the configuration")

    dataset_config = config['dataset']
    hf_name = dataset_config['hf_name']
    split = dataset_config.get('split', None)
    streaming = dataset_config.get('streaming', False)

    dm = DataModule.load_hf(hf_name, split=split, streaming=streaming)

    if 'preprocessor' in dataset_config:
        preprocessor_config = dataset_config['preprocessor']
        match preprocessor_config['type']:
            case 'simple_text':
                column_name = preprocessor_config.get('column_name', 'text')
                max_length = preprocessor_config.get('max_length', None)
                dm = Preprocessor.simple_text(dm, tokenizer, column_name=column_name, max_length=max_length)
            case other:
                raise ValueError(f"Unsupported preprocessor type: {other}")
    return dm

def load_wrapper(config, model):
    if 'wrapper' not in config:
        raise ValueError("Missing [wrapper] section in the configuration")

    wrapper_config = config['wrapper']

    optimizer_config = wrapper_config['optimizer']
    match optimizer_config['name']:
        case "Adalite":
            optimizer = Adalite
        case other:
            if hasattr(optim, other):
                optimizer = getattr(optim, other)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_config['name']}")

    optimizer_args = optimizer_config.get('args', None)

    extra_args = {}
    if optimizer_args is not None:
        extra_args['optimizer_args'] = optimizer_args

    scheduler_name = scheduler=get_hf_scheduler(name=wrapper_config['scheduler']['name'])

    lr = wrapper_config['lr']
    match wrapper_config['type']:
        case "SFTWrapper":
            wrapper = SFTWrapper(
                model=model,
                lr=lr,
                optimizer=optimizer,
                scheduler=get_hf_scheduler(name=wrapper_config['scheduler']['name'],
                **wrapper_config['scheduler'].get('args', {})),
            )
        case "PROWrapper":
            wrapper = PROWrapper(
                model=model,
                lr=lr,
                optimizer=optimizer,
                scheduler=get_hf_scheduler(name=wrapper_config['scheduler']['name'],
                beta=wrapper_config.get('beta', 1.0),
                **wrapper_config['scheduler'].get('args', {}))
            )
        case _:
            raise ValueError(f"Unsupported wrapper type: {wrapper_config['type']}")

    return wrapper

def load_trainer(config):
    if 'trainer' not in config:
        raise ValueError("Missing [trainer] section in the configuration")

    trainer_config = config['trainer']
    max_epochs = trainer_config.get('max_epochs', 1)
    accelerator = trainer_config.get('accelerator', 'gpu')
    devices = trainer_config.get('devices', 1)
    accumulate_grad_batches = trainer_config.get('accumulate_grad_batches', 1)
    log_lr = trainer_config.get('log_lr', True)
    wandb_project = trainer_config.get('wandb_project', None)

    extra_args = {}
    if wandb_project is not None:
        extra_args['logger'] = WandbLogger(project=wandb_project)
    if log_lr:
        extra_args['callbacks']=[LearningRateMonitor(logging_interval='step')]


    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        accumulate_grad_batches=accumulate_grad_batches,
        **extra_args
    )
    
    return trainer

def train_model(config, trainer, model, dm):
    dm.init(batch_size=config['trainer']['batch_size'], collate_fn=DataCollator.Causal())

    trainer.fit(model, dm)

    model_save_path = config.get('model_save_path', 'model_final')
    model.save(model_save_path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.toml>")
        exit(1)

    config = load_config(sys.argv[1])
    model, tokenizer = load_model(config)
    dm = load_dataset(config, tokenizer)
    model = load_wrapper(config, model)
    trainer = load_trainer(config)

    train_model(config, trainer, model, dm)
