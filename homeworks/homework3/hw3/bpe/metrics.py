from __future__ import annotations

from .pretokenizer import *
from .tokenizer import *
from dataclasses import dataclass
from typing import Iterable

def should_div(a: float, b: float) -> float:
    return a / b if b else 0.0

@dataclass
class FreqMetrics:
    
    top_percent: float
    
    unique_words:int
    selected_unique_words: int
    
    tokens_per_word_top_avg: float 
    
@dataclass
class TokenizationMetrics:
    tokens: int
    chars: int
    words: int
    byts: int
    
    tokens_per_word: float
    tokens_per_byte_compress: float
    tokens_per_char_compress: float
    
def get_metrics(tok: BPETokenizer, texts: Iterable[str]) -> TokenizationMetrics:
    tokens = 0
    byts = 0
    chars = 0
    words = 0
    
    for t in texts:
        chars += len(t)
        tokens += len(tok.encode(t))
        byts += len(t.encode("utf-8"))
        words += sum(1 for _ in iter_words(t))
        
    return TokenizationMetrics(
        tokens=tokens,
        chars=chars,
        words=words,
        byts=byts,
        tokens_per_word=should_div(tokens, words),
        tokens_per_byte_compress=should_div(tokens, byts),
        tokens_per_char_compress=should_div(tokens, chars),
    )
    
def get_unused_tokens(
    tok: BPETokenizer,
    texts: Iterable[str],
) -> float:
    # returns fraction
    
    used : set[int] = set()
    for t in texts:
        used.update(tok.encode(t))
    spec = set(tok.spec_token2id.values())
    no_spec_vocab = [i for i in tok.id2token.keys() if i not in spec]
    unused = [i for i in no_spec_vocab if i not in used]
    if len(no_spec_vocab) == 0:
        return 0.0
    return len(unused) / len(no_spec_vocab)
    


def get_tokens_per_word(
    tok: BPETokenizer,
    texts: Iterable[str],
    *,
    top_percent: float = 0.10,
) -> FreqMetrics:
    from collections import Counter
    
    freq: Counter[str] = Counter()
    tokens_per_seen: list[tuple[str,int]] = []
    for t in texts:
        for w in iter_words(t):
            count = len(tok.encode(" "+ w))
            tokens_per_seen += [(w, count)]
            freq[w] += 1
            
    unique = len(freq)
    if unique == 0:
        return FreqMetrics(
            unique_words=0,
            selected_unique_words=0,
            top_percent=top_percent,
            tokens_per_word_top_avg=0.0,
        )
        
    top_tokes, top_seen = 0, 0
    k = max(1, int(round(top_percent * unique)))
    top_words = {w for w, _ in freq.most_common(k)}
    for w, c in tokens_per_seen:
        if w not in top_words:
            continue
        top_tokes += c
        top_seen += 1
    return FreqMetrics(
        unique_words=unique,
        selected_unique_words=len(top_words),
        top_percent=top_percent,
        tokens_per_word_top_avg=should_div(top_tokes, top_seen),
        )
    
