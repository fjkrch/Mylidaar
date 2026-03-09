"""
Download Depth Anything V2 model weights from Hugging Face.
Run this script once before using run_camera.py.

Usage:
    python download_model.py                  # downloads ViT-S (default, ~95MB)
    python download_model.py --encoder vitb   # downloads ViT-B (~390MB)
    python download_model.py --encoder vitl   # downloads ViT-L (~1.3GB)
"""

import argparse
import os
import sys
import urllib.request

# Hugging Face download URLs for each encoder
MODEL_URLS = {
    'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
    'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
    'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
}

MODEL_SIZES = {
    'vits': '~95 MB',
    'vitb': '~390 MB',
    'vitl': '~1.3 GB',
}


def download_with_progress(url, dest):
    """Download a file with a simple progress indicator."""
    print(f"Downloading to {dest} ...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct:3d}%  ({mb_done:.1f} / {mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print("\n  Done!")


def ensure_model(encoder='vits', checkpoints_dir='checkpoints'):
    """Download model weights if they don't already exist. Returns the path."""
    if encoder not in MODEL_URLS:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from: {list(MODEL_URLS.keys())}")

    os.makedirs(checkpoints_dir, exist_ok=True)
    filename = f'depth_anything_v2_{encoder}.pth'
    filepath = os.path.join(checkpoints_dir, filename)

    if os.path.isfile(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Model already exists: {filepath} ({size_mb:.1f} MB)")
        return filepath

    url = MODEL_URLS[encoder]
    print(f"Model not found. Downloading {encoder} ({MODEL_SIZES[encoder]}) ...")
    download_with_progress(url, filepath)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved: {filepath} ({size_mb:.1f} MB)")
    return filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Depth Anything V2 model weights')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'],
                        help='Which encoder to download (default: vits)')
    parser.add_argument('--all', action='store_true', help='Download all available models')
    args = parser.parse_args()

    if args.all:
        for enc in MODEL_URLS:
            ensure_model(enc)
    else:
        ensure_model(args.encoder)
