import torch
from transformers import PreTrainedTokenizer
from typing import Dict
from collections.abc import Sequence

class DataCollator:
    class Causal:
        def __init__(self, pad_id=0):
            self.pad_id = pad_id

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            ids = [d['input_ids'] for d in instances]

            if 'labels' in d[0]:
                labs = [d['labels'] for d in instances]
            else:
                labs = [d['input_ids'] for d in instances]


            max_seq_len = max(len(x) for x in ids)
            pad_len = lambda x: max_seq_len - len(x)

            input_ids = [torch.tensor(x + [self.pad_id]*pad_len(x), dtype=torch.long) for x in ids]
            input_ids = torch.stack(input_ids)
            
            labels = [torch.tensor(x + [-100]*pad_len(x), dtype=torch.long) for x in labs]
            labels = torch.stack(labels)

            attention_mask = [torch.tensor([True]*len(x) + [False]*pad_len(x), dtype=torch.bool) for x in ids]
            attention_mask = torch.stack(attention_mask)

            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )

    class StepByStep:
        def __init__(self, pad_id=0):
            self.pad_id = pad_id

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            ids = []
            labs = []

            for d in instances:
                if 'input_ids' in d and d['input_ids'] is not None:
                    ids.append(d['input_ids'])
                    labs.append(d['labels'])
                else:
                    ids += [d['answer_ids'] + d['rationale_ids']]
                    labs += [d['answer_labels'] + d['rationale_labels']]

            max_seq_len = max(len(x) for x in ids)
            pad_len = lambda x: max_seq_len - len(x)

            input_ids = [torch.tensor(x + [self.pad_id]*pad_len(x), dtype=torch.long) for x in ids]
            input_ids = torch.stack(input_ids)
            
            labels = [torch.tensor(x + [-100]*pad_len(x), dtype=torch.long) for x in labs]
            labels = torch.stack(labels)

            attention_mask = [torch.tensor([True]*len(x) + [False]*pad_len(x), dtype=torch.bool) for x in ids]
            attention_mask = torch.stack(attention_mask)

            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )

    class Ranked:
        def __init__(self, pad_id=0, external_ce_labels=False):
            self.pad_id = pad_id
            self.external_ce_labels=external_ce_labels
            
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


            

            if self.external_ce_labels:
                ce_ids = [x['ce_ids'] for x in batch]
                ce_labs = [x['ce_labels'] for x in batch]

                max_toks = max(len(s) for s in ce_ids)
                
                ce_input_ids = torch.ones((len(ids), max_toks), dtype=torch.long) * self.pad_id
                ce_labels = torch.ones((len(labs), max_toks), dtype=torch.long) * -100
                
                for s, seq in enumerate(ce_ids):
                    for t, tok in enumerate(seq):
                        ce_input_ids[s, t] = tok
                        ce_labels[s, t] = ce_labs[s][t]
                
                return {
                    'input_ids': input_ids,
                    'labels': labels,
                    'ranks': ranks,
                    'attention_mask': attention_mask,
                    'ce_ids': ce_input_ids,
                    'ce_labels': ce_labels
                }


            return {
                'input_ids': input_ids,
                'labels': labels,
                'ranks': ranks,
                'attention_mask': attention_mask
            }
