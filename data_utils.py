import torch
from torch import Tensor, LongTensor
from typing import Tuple, List, Dict


class Tokenizer(): 
    def __init__(self, ) -> None: 
        self.digits = [str(x) for x in range(10)]
        self.operators = ['+', '-']
        self._sos = '<sos>'
        self._eos = '<eos>'
        self._pad = '<pad>'
    
        self.id2sym: List[str] = [self._pad, self._eos, self._sos] + self.digits + self.operators
        self.sym2id: Dict[str, int] = {
            sym: idx for idx, sym in enumerate(self.id2sym)
        }

    @property
    def sos(self): return self.sym2id[self._sos]

    @property
    def eos(self): return self.sym2id[self._eos]

    @property
    def pad(self): return self.sym2id[self._pad]

    @property
    def vocab_size(self,): return len(self.id2sym)

    def __call__(self, exp: str, decoder_input: bool=False) -> List[int]: 
        result = [self.sym2id[s] for s in exp]
        if decoder_input: 
            return [self.sos] + result + [self.eos]
        else: 
            return result
    
    def reverse(self, exp: List[int]) -> str: 
        return ''.join([self.id2sym[idx] for idx in exp])

class ArithmeticDataset(torch.utils.data.Dataset): 
    def __init__(self, metadata: str) -> None: 
        with open(metadata) as f: 
            self.metas: List[Tuple[str, str]] = [l.strip().split('|') for l in f]
        self.tok = Tokenizer()
    
    def __len__(self, ) -> int: return len(self.metas)
    def __getitem__(self, idx): 
        q, a = self.metas[idx]
        return torch.tensor(self.tok(q)), torch.tensor(self.tok(a, decoder_input=True)), (q,a)

class ArithmeticCollate(): 
    def __init__(self, pad_id: int): 
        self.pad_id = pad_id
    
    def pad_tensor(self, ts: List[Tensor]): 
        # t: (T_i, ...)
        maxlen = max(len(t) for t in ts)
        bcsz = len(ts)

        # buffer: (bcsz, max(T), ...)
        buffer = ts[0].new_ones((bcsz, maxlen) + ts[0].shape[1:]) * self.pad_id
        lengths = [len(t) for t in ts]
        for batch_idx, t in enumerate(ts): 
            buffer[batch_idx, :len(t), ...] = t
        
        return buffer, LongTensor(lengths)
    
    def __call__(self, batched): 
        questions, answers, tags = zip(*batched)
        return self.pad_tensor(questions), self.pad_tensor(answers), tags


