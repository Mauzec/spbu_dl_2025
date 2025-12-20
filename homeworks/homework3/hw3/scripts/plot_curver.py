import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from homeworks.homework3.hw3.corpus import pushkin_poems, get_texts
from .download_dataset import iter_text_lines

from homeworks.homework3.hw3.bpe.tokenizer import BPETokenizer
from homeworks.homework3.hw3.bpe.metrics import get_metrics

def train_and_plot_curve(
    file_path: Path,
    pushkin_root: Path,
    max_chars: int,
    output: Path,
    output_trained_root: Path,
    vocab_sizes: list[int],   
) -> None:
    train = get_texts(
        iter_text_lines(
            file_path,
            encoding="utf-8",
            max_chars=max_chars,
        ),
        max_chars=max_chars,
    )
    pushkin = list(pushkin_poems(pushkin_root))
    
    sizes = [int(x.strip()) for x in vocab_sizes.split(",") if x.strip()]
    x,y = [],[]
    for s in sizes:
        tok: BPETokenizer
        if (output_trained_root / f'tok_size_{s}.json').exists():
            tok = BPETokenizer.load(output_trained_root / f'tok_size_{s}.json')
            print(f"cached {f'tok_size_{s}.json'}")
        else:
            tok = BPETokenizer.train(train, size=s)
            output_trained_root.mkdir(parents=True, exist_ok=True)
            tok.save(output_trained_root / f'tok_size_{s}.json')
        
        metr = get_metrics(tok, pushkin)
        x.append(tok.vocab_size)
        y.append(metr.tokens_per_byte_compress)
        print(f'size={tok.vocab_size}, t/b={metr.tokens_per_byte_compress:.5f}')
        
    plt.figure(figsize=(7,4))
    plt.plot(x,y, marker='o')
    plt.xlabel("vocab size")
    plt.ylabel("compression ratio (t/b)")
    plt.title("vocab size & compression ratio")
    plt.grid(True, alpha=.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    print(f'savedto {output}')
    
        

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--files-path", "-fr", type=Path, required=True)
    args.add_argument("--pushkin-root", "-p", type=Path, default=Path("homeworks/homework3/dataverse_files/texts"))
    args.add_argument("--max-chars", "-mc", type=int, default=int(2e6))
    args.add_argument("--output", "-o", type=Path, default=Path("homeworks/homework3/hw3/curver.png"))
    args.add_argument("--output-trained-root", "-otr", type=Path, default=Path("homeworks/homework3/hw3"))
    args.add_argument("--vocab-sizes", "-sizes", type=str, default="1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000")
    pargs = args.parse_args()
    
    train_and_plot_curve(
        file_path=pargs.files_path,
        pushkin_root=pargs.pushkin_root,
        max_chars=pargs.max_chars,
        output=pargs.output,
        output_trained_root=pargs.output_trained_root,
        vocab_sizes=pargs.vocab_sizes,
    )
    
