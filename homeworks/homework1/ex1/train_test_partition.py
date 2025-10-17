import argparse
from pathlib import Path


TRAIN_SPLIT_SIZE = 463_715
TEST_SPLIT_SIZE = 51_630


def partition(
    path: Path,
    train_path: Path, test_path: Path,
    train_size: int = TRAIN_SPLIT_SIZE
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'dataset not here: {path}')
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('r', encoding='utf-8') as source, \
        train_path.open('w', encoding='utf-8') as train_file, \
        test_path.open('w', encoding='utf-8') as test_file:
            
        for idx, line in enumerate(source):
            if idx < train_size:
                train_file.write(line)
            else:
                test_file.write(line)
                
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--dataset', 
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'datasets' / 'YearPredictionMSD.txt',
    )
    p.add_argument(
        '--train',
        type=Path,
        default=Path(__file__).resolve().parent / 'YearPredictionMSD_train.txt',
    ) 
    p.add_argument(
        '--test',
        type=Path,
        default=Path(__file__).resolve().parent / 'YearPredictionMSD_test.txt',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    partition(args.dataset, args.train, args.test)


if __name__ == '__main__': main()
