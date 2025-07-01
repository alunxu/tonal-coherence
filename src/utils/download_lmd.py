#!/usr/bin/env python3
"""
download_lmd.py

Resume download of lmd_matched_h5.tar.gz
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched_h5.tar.gz"
DEST = Path("data/metadata/lmd_matched_h5.tar.gz")

def download_file():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    
    # Check existing size
    resume_header = {}
    mode = 'wb'
    current_size = 0
    
    if DEST.exists():
        current_size = DEST.stat().st_size
        resume_header = {'Range': f'bytes={current_size}-'}
        mode = 'ab'
        print(f"Resuming download from {current_size / 1024 / 1024 / 1024:.2f} GB...")
    else:
        print("Starting new download...")

    try:
        response = requests.get(URL, stream=True, headers=resume_header)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0)) + current_size
        
        block_size = 1024 * 1024 # 1MB
        
        with open(DEST, mode) as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, initial=current_size) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
        print("\n✅ Download complete!")
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")

if __name__ == "__main__":
    download_file()
