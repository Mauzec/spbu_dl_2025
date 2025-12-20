
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset

def _split_multi_dicts(raw: str) -> str | None:
    if raw.count("{") < 2 or raw.count("}") < 2:
        return None
    candidate = re.sub(r"}\s*{", "}, {", raw)
    return f"[{candidate}]"


def _maybe_parse_literal(raw: str) -> Any | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    multi = _split_multi_dicts(raw)
    if multi is not None:
        try:
            return ast.literal_eval(multi)
        except (ValueError, SyntaxError):
            return None
    return None


def _flatten_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = _maybe_parse_literal(value)
        if parsed is not None and not (isinstance(parsed, str) and parsed == value):
            return _flatten_text(parsed)
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "text", "body", "data"):
            if key in value and value[key] is not None:
                nested = _flatten_text(value[key])
                if nested:
                    return nested
        parts = [_flatten_text(v) for v in value.values()]
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else None
    if isinstance(value, list):
        parts = [_flatten_text(item) for item in value]
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else None
    return str(value)


def _get_raw_text(example: Any, text_field: str | None, default_field: str) -> Any | None:
    if isinstance(example, str):
        return example
    if isinstance(example, dict):
        if text_field and text_field in example:
            return example[text_field]
        if default_field in example:
            return example[default_field]
        return example
    if isinstance(example, list):
        return example
    return example

def main() -> None:
    ds = load_dataset(
        "ZeroAgency/ru-big-russian-dataset",
        cache_dir="./ru_big_russian_dataset",
    )
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    split = ds[split_name]
    candidate_fields = [
        "text",
        "content",
        "body",
        "data",
    ]
    text_field = next((name for name in candidate_fields if name in split.column_names), None)
    default_field = split.column_names[0]

    out_path = "ru_corpus.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in split:
            raw = _get_raw_text(ex, text_field, default_field)
            text = _flatten_text(raw)
            if not text:
                continue
            f.write(text.replace("\r\n", "\n").strip() + "\n")

    print(f"Saved to {out_path} from split '{split_name}'")

def iter_text_lines(
    path: Path,
    *,
    encoding: str = "utf-8",
    max_chars: int | None = None,
) -> Iterator[str]:
    used = 0
    with path.open("r", encoding=encoding) as f:
        for raw in f:
            if max_chars is not None and used >= max_chars:
                break
            text = raw.strip()
            if not text:
                continue

            if max_chars is not None:
                remain = max_chars - used
                if len(text) > remain:
                    text = text[:remain]
                used += len(text)
            yield text