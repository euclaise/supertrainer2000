from src.datasets import Preprocessors
from src.wrappers import SFTWrapper

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

model_name = "tiiuae/falcon-rw-1b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset('euirim/goodwiki', split='train')
ds = Preprocessors.simple_text(ds, tokenizer, column_name='markdown')
