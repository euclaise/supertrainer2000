from src.datasets import DataModule, Preprocessors
from src.wrappers import SFTWrapper
from src.utils import get_hf_scheduler, DataCollatorForCausal

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from torch.optim import AdamW

from lightning import Trainer

model_name = "tiiuae/falcon-rw-1b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dm = DataModule.load_hf('euirim/goodwiki', split='train')
dm = Preprocessors.simple_text(dm, tokenizer, column_name='markdown')
dm = dm.truncate_len(256)
dm.init(batch_size = 3, collate_fn = DataCollatorForCausal())

optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_hf_scheduler(name='linear')

wrapper = SFTWrapper(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=1
)

trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1)
trainer.fit(wrapper, dm)

#wrapper.save("model_final")
