from supertrainer2k.datasets import DataModule, Preprocessors
from supertrainer2k.wrappers import SFTWrapper
from supertrainer2k.utils import get_hf_scheduler, DataCollatorForCausal

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from torch.optim import AdamW

from lightning import Trainer
from lightning.pytorch.tuner import Tuner

model_name = "tiiuae/falcon-rw-1b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dm = DataModule.load_hf('euirim/goodwiki', split='train')
dm = Preprocessors.simple_text(dm, tokenizer, column_name='markdown')
dm = dm.truncate_len(256)
dm.init(batch_size = 3, collate_fn = DataCollatorForCausal())

model = SFTWrapper(
    model=model,
    optimizer=AdamW,
    scheduler=get_hf_scheduler(name='linear'),
    batch_size=1,
    lr=1e-5
)

trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1)

# Automatically find the best LR
tuner = Tuner(trainer)
tuner.lr_find(model, dm)

trainer.fit(model, dm)

model.save("model_final")
