from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass
class BPEModel:
    merges: list[tuple[str,str]]
    
    def rank(self) -> dict[tuple[str,str], int]:
        return {pair: i for i, pair in enumerate(self.merges)}
    
    
def train_bpe(
    token_sequences: Iterable[list[str]],
    *,
    target_vocab_size: int,
    # base_symbols: set[str],
    base_symbols_size: int,
    verbose: bool = False,
    verbose_step: int = 50,
) -> BPEModel:
    # input: for ex: 1 sequence: "ab", "bc", "bcd", "cde"
    '''
    Trains BPE merges os a stream of pretokenized pieces.
    Suppose that every piece is already converted to string.
    '''
    
    # build vocab( tuple(symbols): count )
    word_vocab: Counter[tuple[str,...]] = Counter()
    for seq in token_sequences:
        for s in seq:
            if not s: continue
            symbols = tuple(s)
            word_vocab[symbols] += 1
            
            
    merges: list[tuple[str,str]] = []
    
    merges_size = max(0,target_vocab_size-base_symbols_size)
    
    for step in range(merges_size):
        freq: Counter[tuple[str,str]] = Counter()
        for w, f in word_vocab.items():
            for pair in _pairs(w):
                freq[pair] += f
                
        if not freq: break
        
        best_pair, best_count = freq.most_common(1)[0]
        if best_count <= 1: break
        merges.append(best_pair)
        new_vocab: Counter[tuple[str,...]] = Counter()
        for w, f in word_vocab.items():
            new_w = _merge_word(w, best_pair)
            new_vocab[new_w] += f
        word_vocab = new_vocab
        
        if verbose:
            if (step + 1) % 50 == 0:
                print(f'[BPE][{step+1}]\n\tbest_pair={best_pair}\n\tcount={best_count}')
    return BPEModel(merges) 

def encode_bpe(piece: str, bpe_ranks: dict[tuple[str,str], int]) -> list[str]:
    '''
    Apply merges to a single piece. 
    Greedy algorithm.
    '''
    
    if not piece: return []
    
    w = tuple(piece)
    pairs = _pairs(w)
    if not pairs: return [piece]
    
    while 1:
        best = None
        best_rank = None
        for pair in pairs:
            r = bpe_ranks.get(pair)
            
            if r is None: continue
            if best_rank is None or r < best_rank:
                best = pair
                best_rank = r
                
        if best is None: break
        w = _merge_word(w,best)
        pairs = _pairs(w)
        if not pairs: break
        
    return list(w)


def _merge_word(symbols: tuple[str,...], pair: tuple[str,str]) -> tuple[str,...]:
    if len(symbols) <= 1:
        return symbols
    x,y = pair
    merged = []
    
    i: int = 0
    while i < len(symbols):
        if i < len(symbols)-1 and symbols[i] == x and symbols[i+1] == y:
            merged.append(x+y)
            i+=2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def _pairs(symbols: tuple[str,...]) -> set[tuple[str, str]]:
    if len(symbols) <= 1:
        return set()
    return set(zip(symbols, symbols[1:]))