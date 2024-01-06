import torch
from transformers import PreTrainedTokenizer
from typing import Dict
from collections.abc import Sequence

class DataCollator:
    class Causal:
        tokenizer: PreTrainedTokenizer

        def __init__(self, pad_id=0):
            self.pad_id = pad_id

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            ids = [d['input_ids'] for d in instances]
            labels = [d['labels'] for d in instances]

            max_seq_len = max(len(x) for x in ids)

            input_ids = [torch.tensor([self.pad_id]*(max_seq_len - len(x)) + x, dtype=torch.long) for x in ids]
            input_ids = torch.stack(input_ids)
            
            labels = [torch.tensor([-100]*(max_seq_len - len(x)) + x, dtype=torch.long) for x in labels]
            labels = torch.stack(labels)

            attention_mask = [torch.tensor([False]*(max_seq_len - len(x)) + [True]*len(x), dtype=torch.bool) for x in ids]
            attention_mask = torch.stack(attention_mask)

            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )

    class Ranked:
        def __init__(self, pad_id=0):
            self.pad_id = pad_id
            
        def __call__(self, batch):
            ids = [x['input_ids'] for x in batch]
            labs = [x['labels'] for x in batch]
            rankl = [x['ranks'] for x in batch]

        
            max_toks = max(max(len(y) for y in x) for x in ids)
            max_seqs = max(len(x) for x in ids)
                        
            input_ids = torch.ones((len(ids), max_seqs, max_toks), dtype=torch.long) * self.pad_id
            labels = torch.ones((len(labs), max_seqs, max_toks), dtype=torch.long) * -100
            ranks = torch.ones((len(ids), max_seqs), dtype=torch.long) * -100
            attention_mask = torch.zeros((len(ids), max_seqs, max_toks), dtype=torch.bool)

            for b, q in enumerate(ids):
                for s, seq in enumerate(q):
                    for t, tok in enumerate(seq):
                        input_ids[b, s, t] = tok
                        labels[b, s, t] = labs[b][s][t]
                        attention_mask[b, s, t] = True
                    ranks[b, s] = rankl[b][s]

            return {
                'input_ids': input_ids,
                'labels': labels,
                'ranks': ranks,
                'attention_mask': attention_mask
            }
