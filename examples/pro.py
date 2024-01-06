from supertrainer2k.datasets import DataModule, Preprocessor, DataCollator
from supertrainer2k.wrappers import PROWrapper
from supertrainer2k.optim import get_hf_scheduler

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


dm = Preprocessor.multi_choice_text(dm, tokenizer, max_length=512)
dm.init(batch_size = 1, collate_fn = DataCollator.Ranked())

model = PROWrapper(
    model=model,
    optimizer=AdamW,
    scheduler=get_hf_scheduler(name='linear'),
    lr=2e-5
)

trainer = Trainer(max_epochs=1, accelerator="gpu", accumulate_grad_batches=2)

trainer.fit(model, dm)
model.save("model_final")
