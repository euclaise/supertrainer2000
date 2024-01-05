from supertrainer2k.datasets import DataModule, Preprocessors, DataCollator
from supertrainer2k.wrappers import PROWrapper
from supertrainer2k.utils import get_hf_scheduler

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from torch.optim import AdamW

from lightning import Trainer
from lightning.pytorch.tuner import Tuner

model_name = "tiiuae/falcon-rw-1b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dm = DataModule.load_hf('euclaise/SuperMC')


def apply_template(row):
    chosen = "Q: " + row['msg'] + '\n\nA: ' + row['resp_correct']
    rejected = ["Q: " + row['msg'] + '\n\nA: ' + r for r in row['resp_incorrect']]

    return {'chosen': chosen, 'rejected': rejected}
dm = dm.map(apply_template)


dm = Preprocessors.multi_choice_text(dm, tokenizer, tokenizer_kwargs = {'truncation': True, 'max_length': 128})
dm.init(batch_size = 1, collate_fn = DataCollator.MultiChoice())

model = PROWrapper(
    model=model,
    optimizer=AdamW,
    scheduler=get_hf_scheduler(name='linear'),
    batch_size=1,
    lr=1e-5
)

trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1)

trainer.fit(model, dm)

model.save("model_final")