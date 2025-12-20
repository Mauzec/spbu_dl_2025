from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator
import re

@dataclass
class Pretokenizer:
    pat: re.Pattern
    
    @classmethod
    def build(cls) -> Pretokenizer:
        letters = "A-Za-z" + "А-Яа-яЁё"
        digits = "0-9"
        return cls(pat=re.compile(
            r"<[^>]+>"
            r"| ?[" + letters + r"]+" # letters
            r"| ?[0-9]+" # digits
            r"| ?[^\s" + letters + digits + r"]+" # other
            r"|\s+(?!\S)" # trailing spaces
            r"|\s+" # other spaces
        ))
    
    def split(self, text: str) -> list[str]:
        return self.pat.findall(text)
    
    def iter_split(self, text: str) -> Iterator[str]:
        yield from self.pat.findall(text)
        

_word = re.compile(r"[" + "A-Za-z" + "А-Яа-яЁё" + r"" + "0-9" + r"]+")
def iter_words(text: str) -> Iterator[str]:
    yield from _word.findall(text)

