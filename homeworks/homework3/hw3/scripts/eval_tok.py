from pathlib import Path

from homeworks.homework3.hw3.bpe.pretokenizer import Pretokenizer
from homeworks.homework3.hw3.bpe.tokenizer import BPETokenizer
from homeworks.homework3.hw3.bpe.bytes import ByteUnicodeEncoder as BUE
from homeworks.homework3.hw3.bpe.bpe import encode_bpe
from homeworks.homework3.hw3.corpus import local_docs_code, pushkin_poems
from homeworks.homework3.hw3.bpe.metrics import get_metrics, get_tokens_per_word, get_unused_tokens

import argparse

def eval(
    tok_path: Path,
    files_root: Path,
    pushkin: Path,
) -> None:
    tok = BPETokenizer.load(tok_path)
    local = list(local_docs_code(files_root))
    texts = list(pushkin_poems(pushkin))

    metrics_docs = get_metrics(tok, local)
    metrics_pushkin = get_metrics(tok, texts)
    print("docs + code")
    print(f"tokens={metrics_docs.tokens} bytes={metrics_docs.byts} chars={metrics_docs.chars} words={metrics_docs.words}")
    print(f"compress t/b:{metrics_docs.tokens_per_byte_compress:.5f}")    
    print(f"compress t/c:{metrics_docs.tokens_per_char_compress:.5f}")
    print(f"avg t/w:{metrics_docs.tokens_per_word:.5f}")
    print("pushkin")
    print(f"tokens={metrics_pushkin.tokens} bytes={metrics_pushkin.byts} chars={metrics_pushkin.chars} words={metrics_pushkin.words}")
    print(f"compress t/b:{metrics_pushkin.tokens_per_byte_compress:.5f}")    
    print(f"compress t/c:{metrics_pushkin.tokens_per_char_compress:.5f}")
    print(f"avg t/w:{metrics_pushkin.tokens_per_word:.5f}")
    print()
    top_local = get_tokens_per_word(tok, local, top_percent=.16)
    top_pushkin = get_tokens_per_word(tok, texts, top_percent=.16)
    unused_pushkin = get_unused_tokens(tok, texts)
    print(f"docs+code: avg tokens/w_top16%={top_local.tokens_per_word_top_avg:.5f}, unique={top_local.unique_words}, selected={top_local.selected_unique_words}")
    print(f"pushkin: avg tokens/w_top16%={top_pushkin.tokens_per_word_top_avg:.5f}, unique={top_pushkin.unique_words}, selected={top_pushkin.selected_unique_words}")
    print(f"pushkin: unused token fraction={unused_pushkin:.5f}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--tokenizer", type=Path, required=True)
    args.add_argument("--files-root", "-fr", type=Path, required=True)
    args.add_argument("--pushkin", "-p", type=Path, default=Path("homeworks/homework3/dataverse_files/texts"))
    pargs = args.parse_args()
    
    eval(
        tok_path=pargs.tokenizer,
        files_root=pargs.files_root,
        pushkin=pargs.pushkin,
    )