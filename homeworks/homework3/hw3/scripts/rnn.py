import argparse
from pathlib import Path
from typing import Iterable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from homeworks.homework3.hw3.bpe.tokenizer import BPETokenizer
from homeworks.homework3.hw3.scripts.download_dataset import iter_text_lines

import time

class LMDataset(Dataset):
    def __init__(
        self,
        seqs: list[list[int]],
        block_size: int,
    ) -> None:
        self.data = []
        for s in seqs:
            
            # cut to blocks with eos and bos
            for i in range(0, len(s)-1, block_size):
                chunk = s[i:i+block_size+1]
                if len(chunk) < 2: continue
                self.data += [chunk]
                
        
        if not self.data:
            raise ValueError("no chunks; check block_size or data")
        
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, index):
        s = self.data[index]
        x = torch.tensor(s[:-1], dtype=torch.long)
        y = torch.tensor(s[1:], dtype=torch.long)
        return x, y
    
def pack_batch(batch):
    xs,ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    for i, (x,y) in enumerate(zip(xs,ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y
    return x_pad, y_pad


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embdim: int = 256,
        hidden: int = 512,
        num_layers: int = 2,
        dropout: float = .15
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embdim)
        self.rnn = nn.GRU(
            embdim,
            hidden,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden, vocab_size)
    
    # def
    
    def forward(self, x):
        out, _ = self.rnn(self.emb(x))
        return self.proj(out)
    

def load_sequences(
    tok: BPETokenizer,
    path: Path,
    max_chars: int,
    add_spec: bool = True,
)-> list[list[int]]: 
    seqs = []
    for line in iter_text_lines(path, max_chars=max_chars):
        ids = tok.encode(line, add_spec=add_spec)
        if len(ids) > 1:
            seqs.append(ids)
    return seqs

def train(
    tok_path: Path,
    corpus_path: Path,
    *,
    max_chars: int,
    
    block_size: int,
    batch_size: int,
    
    lr: float,
    epochs:int,
    device: str,
) -> None:
    tok = BPETokenizer.load(tok_path)
    seqs = load_sequences(
        tok,
        corpus_path,
        max_chars=max_chars,
        add_spec=True,
    )
    ds = LMDataset(seqs, block_size=block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=pack_batch, num_workers=5)
    
    model = RNN(tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    time0 = time.time()
    for epoch in range(1, epochs+1):
        total_loss = .0
        total_tokens = 0
        for x, y in loader:
            x: torch.Tensor = x.to(device)
            y: torch.Tensor = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            
        ppl = torch.exp(torch.tensor(total_loss / total_tokens) if total_tokens else torch.tensor(float('inf')))
        time1 = time.time()
        print(f'[{time1 - time0:.1f}s] epoch={epoch} loss={total_loss/total_tokens:.4f} ppl={ppl:.4f}')
        
    torch.save(model.state_dict(), tok_path.with_name("rnn.pt"))
    print(f'saved weights to {tok_path.with_name("rnn.pt")}')
    
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--tokenizer-path", "-tp", type=Path, required=True)
    args.add_argument("--corpus-path", "-cp", type=Path, required=True)
    args.add_argument("--max-chars", "-mc", type=int, default=int(4e6))
    args.add_argument("--block-size", "-bs", type=int, default=128)
    args.add_argument("--batch-size", "-b", type=int, default=32)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--epochs", "-e", type=int, default=9)
    args.add_argument("--device", "-d", type=str, default="cuda")
    return args.parse_args()
    
    
if __name__ == "__main__":
    args = parse_args()
    print("device:", args.device)
    
    train(
        tok_path=args.tokenizer_path,
        corpus_path=args.corpus_path,
        max_chars=args.max_chars,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )
        