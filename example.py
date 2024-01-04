from src.datasets import DataModule, Preprocessors
from src.wrappers import SFTWrapper

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from torch.optim import AdamW

model_name = "tiiuae/falcon-rw-1b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dm = DataModule.load_hf('euirim/goodwiki', split='train')
dm = Preprocessors.simple_text(dm, tokenizer, column_name='markdown')

optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_hf_scheduler(name='linear')

wrapper = SFTWrapper(
    model=model,
    data_module=dm,
    optimizer=optimizer,
    scheduler=scheduler
)

trainer = Trainer(max_epochs=1, gpus=1)
trainer.fit(wrapper)

wrapper.save("model_final")
