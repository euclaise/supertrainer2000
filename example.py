from src.datasets import DataModule, Preprocessors
from src.wrappers import SFTWrapper

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "tiiuae/falcon-rw-1b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dm = DataModule.load_hf('euirim/goodwiki', split='train')
dm = Preprocessors.simple_text(dm, tokenizer, column_name='markdown')
print(dm)
