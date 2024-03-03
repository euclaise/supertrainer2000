import multiprocessing as mp
import torch

NUM_PROC = max(1, mp.cpu_count() - 1)
torch.set_float32_matmul_precision('medium')
