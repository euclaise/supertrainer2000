import multiprocessing as mp

NUM_PROC = max(1, mp.cpu_count() - 1)
