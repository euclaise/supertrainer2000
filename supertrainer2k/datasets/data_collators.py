import torch
from transformers import PreTrainedTokenizer
from typing import Dict
from collections.abc import Sequence

class DataCollatorForCausal:
    tokenizer: PreTrainedTokenizer

    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        toks = [d['toks'] for d in instances]

        max_seq_len = max(len(x) for x in toks)

        input_ids = [torch.tensor([self.pad_id]*(max_seq_len - len(x)) + x, dtype=torch.long) for x in toks]
        input_ids = torch.stack(input_ids)
        
        labels = [torch.tensor([-100]*(max_seq_len - len(x)) + x, dtype=torch.long) for x in toks]
        labels = torch.stack(labels)

        attention_mask = [torch.tensor([False]*(max_seq_len - len(x)) + [True]*len(x), dtype=torch.bool) for x in toks]
        attention_mask = torch.stack(attention_mask)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )

    
class DataCollatorForMultiChoice:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id
        
    def __call__(self, batch):
        bsz = len(batch)
        seqs = [[x['toks_chosen']] + x['toks_rejected'] for x in batch]
    
        max_toks = max(max(len(y) for y in x) for x in seqs)
        max_seqs = max(len(x) for x in seqs)
                    
        input_ids = torch.ones((len(seqs), max_seqs, max_toks), dtype=torch.long) * self.pad_id
        labels = torch.ones((len(seqs), max_seqs, max_toks), dtype=torch.long) * -100
        mask = torch.zeros((len(seqs), max_seqs), dtype=torch.bool)

        for b, q in enumerate(seqs):
            for s, seq in enumerate(q):
                for t, tok in enumerate(seq):
                    input_ids[b, s, t] = tok
                    labels[b, s, t] = tok
                mask[b, s] = 1

        return {
            'input_ids': input_ids,
            'labels': labels,
            'mask': mask
        }
