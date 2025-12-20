from pathlib import Path

from homeworks.homework3.hw3.bpe.pretokenizer import Pretokenizer
from homeworks.homework3.hw3.bpe.tokenizer import BPETokenizer
from homeworks.homework3.hw3.bpe.bytes import ByteUnicodeEncoder as BUE
from homeworks.homework3.hw3.bpe.bpe import encode_bpe
from homeworks.homework3.hw3.corpus import get_texts
from .download_dataset import iter_text_lines
import argparse


def train_tok(
    file_path: Path,
    vocab_size: int,
    max_chars: int,
    output: Path,
    verbose: bool,
) -> None:
    texts = get_texts(
        iter_text_lines(
            file_path,
            encoding="utf-8",
            max_chars=max_chars,
        ),
        max_chars=max_chars,
    )
    tok = BPETokenizer.train(
        texts,
        verbose=verbose,
        size=vocab_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tok.save(output)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # args.add_argument("--files-root", "-fr", type=Path, required=True)
    args.add_argument("--file-path", "-fp", type=Path, required=True)
    args.add_argument("--vocab-size", "-size", type=int, default=4000)
    args.add_argument("--max-chars", "-mc", type=int, default=int(2e6))
    args.add_argument("--output", "-o", type=Path, default=Path("homeworks/homework3/hw3/tok.json"))
    args.add_argument("--verbose", "-v", action="store_true")
    pargs = args.parse_args()
    
    train_tok(
        file_path=pargs.file_path,
        vocab_size=pargs.vocab_size,
        max_chars=pargs.max_chars,
        output=pargs.output,
        verbose=pargs.verbose,
    )
    