from PIL import Image
import os
from pathlib import Path

DATA_PATH = Path("./one-piece-classification/splitted")
TRAIN_PATH = Path(DATA_PATH / "train") # 
TEST_PATH = Path(DATA_PATH / "test")

def convert_folder(path: Path):
    for p in path.rglob("*.png"):
        with Image.open(p) as img:
            if img.mode in ("P", "LA") or (img.mode == "RGBA" and "transparency" in img.info):
                img.convert("RGB").save(p)
convert_folder(TRAIN_PATH)
convert_folder(TEST_PATH)