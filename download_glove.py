"""
download_glove.py
=================
Run this script on the HPC login node BEFORE submitting run.slurm.
Compute nodes typically block outbound internet, so the file must be
downloaded in advance.

Usage
-----
    python download_glove.py
    python download_glove.py --path /scratch/myuser/glove.6B.300d.txt
"""

import argparse
import os
import urllib.request
import zipfile


def download_glove(path='glove.6B.300d.txt'):
    if os.path.isfile(path):
        print(f"GloVe file already exists: {path}")
        return

    url      = 'https://nlp.stanford.edu/data/glove.6B.zip'
    zip_file = path.replace('glove.6B.300d.txt', 'glove.6B.zip') if path != 'glove.6B.300d.txt' else 'glove.6B.zip'

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  Downloading GloVe: {pct:.1f}%  ({downloaded/1e6:.0f} MB)",
                  end='', flush=True)

    print(f"Downloading GloVe vectors from Stanford (~822 MB) -> {path}")
    urllib.request.urlretrieve(url, zip_file, reporthook=_progress)
    print()

    print(f"Extracting glove.6B.300d.txt ...")
    with zipfile.ZipFile(zip_file) as z:
        z.extract('glove.6B.300d.txt', path=os.path.dirname(path) or '.')
    os.remove(zip_file)

    # rename if a custom path was requested
    extracted = os.path.join(os.path.dirname(path) or '.', 'glove.6B.300d.txt')
    if os.path.abspath(extracted) != os.path.abspath(path):
        os.rename(extracted, path)

    print(f"GloVe ready -> {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='glove.6B.300d.txt',
                        help='Destination file path (default: glove.6B.300d.txt)')
    args = parser.parse_args()
    download_glove(args.path)
