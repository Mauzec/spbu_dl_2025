from __future__ import annotations
from dataclasses import dataclass

def bytes_to_unicode():
    """
    LICENSE: OpenAI's GPT-2 (https://github.com/openai/gpt-2/blob/master/src/encoder.py)
    FROM: https://github.com/markhliu/DGAI/blob/main/utils/bpe.py
    
    Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
    character that represents it visually. Some bytes have their appearance preserved
    because they don't cause any trouble. These are defined in list bs. For example:
    chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
    However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these
    bytes, into new characters in a range where chr() returns a single nice character.
    So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
    In particular, the space character is 32, which we can see by ord(' '). Instead,
    this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
    So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters
    that "look nice", either in their original form, or a funny shifted character
    like 'Ā', or 'Ġ', etc.
    """
    # the 188 integers that render fine in their original form and need no shifting
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+\
        list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:] # all integers b in bs will simply map to chr(b) in the output dict
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    n = 0
    for b in range(2**8):
        if b not in bs:
            # if this byte is "ugly" then map it to the next available "nice" character
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return d

def unicode_to_bytes() -> dict[str, int]:
    u = bytes_to_unicode()
    return {v: k for k, v in u.items()}

@dataclass
class ByteUnicodeEncoder:
    bytes2unicode: dict[int,str]
    unicode2bytes: dict[str,int]
    @classmethod
    def build(cls) -> ByteUnicodeEncoder:
        b2u = bytes_to_unicode()
        u2b = unicode_to_bytes()
        return cls(bytes2unicode=b2u, unicode2bytes=u2b)
    
    def encode(self, bs: bytes) -> str:
        return ''.join(self.bytes2unicode[b] for b in bs)
    def decode(self, s: str) -> bytes:
        return bytes(self.unicode2bytes[ch] for ch in s)
    