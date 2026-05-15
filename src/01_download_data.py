import argparse
from pathlib import Path
import shutil
import kagglehub

from utils import ensure_dirs

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='snap/amazon-fine-food-reviews')
    parser.add_argument('--output',default='data/raw/Reviews.csv')
    args=parser.parse_args()
    out=Path(args.output)
    ensure_dirs(out.parent)
    if out.exists():
        print(f'[INFO] Found local file: {out}. Skip download.')
        return
    print('[INFO] Downloading dataset via kagglehub...')
    path=Path(kagglehub.dataset_download(args.dataset))
    src=path/'Reviews.csv'
    if not src.exists():
        raise FileNotFoundError(f'Reviews.csv not found in {path}')
    shutil.copy2(src,out)
    print(f'[INFO] Saved to {out}')

if __name__=='__main__':
    main()
