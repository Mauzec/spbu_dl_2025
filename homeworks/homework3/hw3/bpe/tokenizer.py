from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from typing import Iterator
from pathlib import Path

from .bytes import ByteUnicodeEncoder
from .pretokenizer import Pretokenizer
from .bpe import BPEModel, train_bpe, encode_bpe

import json

@dataclass
class BPETokenizerConfig:
    spec_tokens: tuple[str, ...] = (
        # <unk> always first, dont remove it
        "<unk>",  # for unknown symbols that are not in vocab
        "<pad>",  # for padding
        "<bos>",  # begin of seq
        "<eos>",  # end of seq
    )
    
    def __post_init__(self):
        if "<unk>" not in self.spec_tokens:
            raise ValueError("<unk> token must be first in spec_tokens")


class BPETokenizer:
    def __init__(
        self,
        *,
        cfg: BPETokenizerConfig | None = None,
        bpe_model: BPEModel | None = None,
        byte_unicode: ByteUnicodeEncoder | None = None,
        vocab: dict[str, int] | None = None,
        spec_token2id: dict[str, int] | None = None,
    ) -> None:
        self.cfg = cfg or BPETokenizerConfig()
        self.bpe_model = bpe_model or BPEModel(merges=[])
        self.byte_unicode = byte_unicode or ByteUnicodeEncoder.build()

        self.pretok = Pretokenizer.build()
        self._bpe_ranks = self.bpe_model.rank()

        self.vocab = vocab or {}
        self.spec_token2id = spec_token2id or {}
        self.id2token = {i: t for t, i in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_vocab(self) -> None:
        # base symbols
        base = set(self.byte_unicode.bytes2unicode.values())

        # let speacial tokens be first
        vocab: dict[str, int] = {}
        spec_token2id: dict[str, int] = {}
        for tok in self.cfg.spec_tokens:
            if tok in vocab:
                continue
            spec_token2id[tok] = len(vocab)
            vocab[tok] = len(vocab)

        # all base symbols
        for s in sorted(base):
            if s in vocab:
                continue
            vocab[s] = len(vocab)
        # all bpe merges
        for x, y in self.bpe_model.merges:
            merged = x + y
            if merged in vocab:
                continue
            vocab[merged] = len(vocab)

        self.vocab = vocab
        self.spec_token2id = spec_token2id
        self.id2token = {i: t for t, i in self.vocab.items()}

    def encode(
        self,
        s: str,
        *,
        add_spec: bool = False,
    ) -> list[int]:
        tokens: list[int] =[]
        if add_spec:
            if "<bos>" in self.spec_token2id:
                tokens += [self.spec_token2id["<bos>"]]
        
        ps = list(self.pretok.iter_split(s))
        for p in ps:
            if p == "": continue
            if p in self.spec_token2id:
                # print("spec:", p)
                tokens += [self.spec_token2id[p]]
                continue
            
            mp = self.byte_unicode.encode(
                p.encode("utf-8")
            )
            
            symbols = encode_bpe(mp, self._bpe_ranks)
            for sym in symbols:
                token_id = self.vocab.get(
                    sym,
                    self.spec_token2id.get("<unk>"),
                )
                if token_id is None:
                    raise KeyError("Uknown token, <unk> not found")
                tokens.append(token_id)
        if add_spec:
            if "<eos>" in self.spec_token2id:
                tokens += [self.spec_token2id["<eos>"]]
        return tokens
    
    def decode(
        self,
        ids: Iterable[int],
        *,
        skip_spec:bool = True,
    ) ->str:
        res = []
        
        for token_id in ids:
            token = self.id2token.get(int(token_id))
            if token is None: continue
            if skip_spec and token in self.spec_token2id:
                continue
            res.append(token)
            

        merged = "".join(res)
        return self.byte_unicode.decode(merged).decode("utf-8", errors="replace")
            
    

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        size: int,
        cfg: BPETokenizerConfig | None = None,
        verbose: bool = False,
        verbose_step: int = 50,
    ) -> "BPETokenizer":
        cfg = cfg or BPETokenizerConfig()

        tok = cls(cfg=cfg)
        base = set(tok.byte_unicode.bytes2unicode.values())

        # merge
        def iter_seq() -> Iterator[list[str]]:
            for t in texts:
                ps = list(tok.pretok.iter_split(t))
                m = [tok.byte_unicode.encode(p.encode("utf-8")) for p in ps]
                yield m

        size = max(size, len(cfg.spec_tokens) + len(base))
        bpe_model = train_bpe(
            iter_seq(),
            target_vocab_size=size - len(cfg.spec_tokens),
            base_symbols_size=len(base),
            verbose=verbose,
            verbose_step=verbose_step,
        )

        tok.bpe_model = bpe_model
        tok._bpe_ranks = bpe_model.rank()

        tok._build_vocab()
        return tok

    @classmethod
    def load(cls, path: Path | str) -> "BPETokenizer":
        path = Path(path)
        p: dict = json.loads(path.read_text(encoding="utf-8"))
        cfg = p.get("cfg", {})
        cfg = BPETokenizerConfig(
            spec_tokens=tuple(
                cfg.get("spec_tokens", ["<pad>", "<unk>", "<bos>", "<eos>"])
            )
        )
        bpe_model = BPEModel(
            merges=[tuple(m) for m in p.get("merges", [])]
        )
        
        tok = cls(
            cfg=cfg,
            bpe_model=bpe_model,
            vocab=p.get("vocab", {}),
        )
        
        tok._bpe_ranks = bpe_model.rank()
        tok.spec_token2id = {
            t: tok.vocab[t] for t in cfg.spec_tokens if t in tok.vocab
        }
        tok.id2token = {i: t for t, i in tok.vocab.items()}
        return tok

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(
                {
                    "merges": [list(m) for m in self.bpe_model.merges],
                    "vocab": self.vocab,
                    "cfg": {
                        "spec_tokens": list(self.cfg.spec_tokens)
                    },
                },
                ensure_ascii=False,
                indent=4,
            ),
            encoding="utf-8",
        )
