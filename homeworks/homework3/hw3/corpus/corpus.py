from typing import Iterator, Iterable, Set
from pathlib import Path
from dataclasses import dataclass

MB2 = 2*1024*1024

def pushkin_poems(path: Path, errors: str = "ignore") -> Iterator[str]:
    """
    """
    
    for p in sorted(path.glob("*.txt")):
        if not p.is_file():
            continue
        try:
            yield p.read_text(encoding="utf-8")
        except Exception as e:
            yield p.read_text(encoding="utf-8", errors=errors)


def iter_files(
    paths: Iterable[Path],
    *, 
    type: str = "utf-8", 
    errors: str = "ignore",
) -> Iterator[str]:
    def file_i(p: Path) -> Iterator[str]:
        try:
            yield p.read_text(encoding=type)
        except UnicodeDecodeError:
            yield p.read_text(encoding=type, errors=errors)
        except Exception as e:
            print(f"[iter_files]: error reading file {p}: {e}")
    for p in paths:
        if not p.is_dir():
            yield from file_i(p)
        else:
            for s in sorted(p.rglob("*")):
                if not s.is_file():
                    continue
                yield from file_i(s)
            

            
def local_docs_code(path: Path, errors: str = "ignore") -> Iterator[str]:
    files: list[Path] = []
    for s in [
        "README.md",
        "**/*.py",
    ]: 
        files.extend(sorted(path.glob(s)))
        
    res: list[Path]=[]
    for p in files:
        if not p.is_file() or p.stat().st_size > MB2:
            continue
        if p.suffix.lower() not in {".md", ".py"}:
            print("[localh-docs-code]: check regex filter, file=", p)
            continue
        res.append(p)
    yield from iter_files(res, errors=errors)
        
    
def get_texts(
    texts: Iterable[Path],
    *,
    max_chars: int | None = None,
) -> list[str]:
    if max_chars is None: return list(texts)
    res = []
    total = 0
    for t in texts:
        if total >= max_chars:
            break
        batch = t[: max_chars - total]
        res.append(batch)
        total += len(batch)
    return res