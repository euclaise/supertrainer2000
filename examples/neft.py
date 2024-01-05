from supertrainer2k.datasets import DataModule, Preprocessor, DataCollator
from supertrainer2k.wrappers import SFTWrapper
from supertrainer2k.optim import get_hf_scheduler, Adalite
from supertrainer2k.misc import apply_neft, remove_neft

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from lightning import Trainer
from lightning.pytorch.tuner import Tuner

model_name = "tiiuae/falcon-rw-1b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# supertrainer2000 supports streaming, making it much easier to handle massive datasets
dm = DataModule.load_hf('JeanKaddour/minipile', split='train', streaming=True)

dm = Preprocessor.simple_text(dm, tokenizer, column_name='text', max_length=256)
dm.init(batch_size = 3, collate_fn = DataCollator.Causal())

model = SFTWrapper(
    model=model,
    optimizer=Adalite,
    scheduler=get_hf_scheduler(name='linear'),
    lr=1e-5
)
apply_neft(model, alpha=5, seq_len=256)


trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1)


trainer.fit(model, dm)
remove_neft(model)
model.save("model_final")
