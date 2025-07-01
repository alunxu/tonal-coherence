#!/usr/bin/env python3
"""
filter_lmd.py

Filter LMD dataset to:
1. Remove noisy/low-quality MIDI files (quality filters)
2. Remove classical music contamination (genre filtering)
3. Balance genres for fair comparison with DLC

Output: Filtered distribution files ready for TDM analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import json
from scipy.stats import entropy
from collections import Counter

from src.utils.loaders import RESULTS_DIR

# --- Quality Filter Thresholds ---
QUALITY_FILTERS = {
    'min_unique_pitches': 5,        # At least 5 non-zero pitch classes
    'min_pitch_entropy': 1.5,       # Entropy in bits (avoid monotonic)
    'max_single_pitch_ratio': 0.5,  # No single pitch > 50%
}

# --- Genre Configuration ---
# Genres to EXCLUDE (classical-adjacent or inappropriate for pop comparison)
EXCLUDED_GENRES = {'Classical', 'Jazz', 'New Age', 'Blues'}

# Target sample sizes per genre (for balanced sampling)
GENRE_TARGETS = {
    'Rock': 400,
    'Pop': 400,
    'Country': 200,
    'Electronic': 200,
    'RnB': 200,
    'Metal': 150,
    'Rap': 150,
    'Latin': 100,
    'Reggae': 100,
    'Folk': 50,
    'World': 50,
    'Punk': 50,
}


def compute_quality_metrics(dist):
    """Compute quality metrics from a pitch distribution."""
    if dist.sum() == 0:
        return None
    
    dist_norm = dist / dist.sum()
    
    return {
        'unique_pitches': np.sum(dist > 0),
        'entropy': entropy(dist_norm + 1e-10, base=2),
        'max_ratio': np.max(dist_norm),
    }


def passes_quality_filters(metrics):
    """Check if metrics pass quality thresholds."""
    if metrics is None:
        return False
    
    return (
        metrics['unique_pitches'] >= QUALITY_FILTERS['min_unique_pitches'] and
        metrics['entropy'] >= QUALITY_FILTERS['min_pitch_entropy'] and
        metrics['max_ratio'] <= QUALITY_FILTERS['max_single_pitch_ratio']
    )


def load_and_filter():
    """Load LMD data and apply all filters."""
    print("="*70)
    print("LMD DATASET FILTERING (with Genre)")
    print("="*70)
    
    # --- Load distributions ---
    print("\nLoading distributions...")
    lmd_35d = np.load(RESULTS_DIR / "data" / "lmd_pitch_class_distributions_35d_partitura.npz", allow_pickle=True)
    distributions = lmd_35d['distributions']
    piece_ids = lmd_35d['piece_ids'].tolist()
    print(f"  Original: {len(piece_ids)} pieces")
    
    # --- Load genre mapping ---
    print("\nLoading genre mapping...")
    genre_path = RESULTS_DIR / "data" / "lmd_genre_mapping.json"
    if not genre_path.exists():
        raise FileNotFoundError(f"Genre mapping not found: {genre_path}\nRun the genre mapping script first.")
    
    with open(genre_path) as f:
        md5_to_genre = json.load(f)
    print(f"  Genre mappings: {len(md5_to_genre)}")
    
    # --- STEP 1: Quality + Genre Filtering ---
    print("\n[1/2] Applying Quality + Genre Filters...")
    
    quality_fail = 0
    no_genre = 0
    excluded_genre = 0
    
    filtered_data = []  # (index, md5, genre)
    
    for i, (dist, pid) in enumerate(zip(distributions, piece_ids)):
        md5 = pid.replace('.mid', '')
        
        # Quality check
        metrics = compute_quality_metrics(dist)
        if not passes_quality_filters(metrics):
            quality_fail += 1
            continue
        
        # Genre check
        if md5 not in md5_to_genre:
            no_genre += 1
            continue
        
        genre = md5_to_genre[md5]
        if genre in EXCLUDED_GENRES:
            excluded_genre += 1
            continue
        
        filtered_data.append((i, md5, genre))
    
    print(f"  Quality fail: {quality_fail}")
    print(f"  No genre: {no_genre}")
    print(f"  Excluded genre (Jazz/Classical/etc): {excluded_genre}")
    print(f"  Pass all filters: {len(filtered_data)}")
    
    # Genre breakdown
    genre_counts = Counter([x[2] for x in filtered_data])
    print(f"\n  Genre breakdown:")
    for genre, count in genre_counts.most_common():
        print(f"    {genre}: {count}")
    
    # --- STEP 2: Genre-Balanced Sampling ---
    print("\n[2/2] Genre-Balanced Sampling...")
    
    # Group by genre
    by_genre = {}
    for idx, md5, genre in filtered_data:
        if genre not in by_genre:
            by_genre[genre] = []
        by_genre[genre].append((idx, md5, genre))
    
    # Sample according to targets
    np.random.seed(42)
    final_data = []
    
    for genre, target in GENRE_TARGETS.items():
        if genre in by_genre:
            items = by_genre[genre]
            n = min(len(items), target)
            sampled = np.random.choice(len(items), n, replace=False)
            for s in sampled:
                final_data.append(items[s])
            print(f"  {genre}: {n} / {target} (available: {len(items)})")
    
    # Add remaining genres not in targets
    for genre, items in by_genre.items():
        if genre not in GENRE_TARGETS:
            n = min(len(items), 50)
            sampled = np.random.choice(len(items), n, replace=False)
            for s in sampled:
                final_data.append(items[s])
            print(f"  {genre} (other): {n}")
    
    print(f"\n  Total sampled: {len(final_data)}")
    
    # --- Save ---
    print("\nSaving filtered data...")
    
    final_indices = [x[0] for x in final_data]
    final_genres = [x[2] for x in final_data]
    final_dists = distributions[final_indices]
    final_pids = [piece_ids[i] for i in final_indices]
    
    output_path = RESULTS_DIR / "data" / "lmd_filtered_with_genre.npz"
    np.savez(
        output_path,
        distributions=final_dists,
        piece_ids=np.array(final_pids),
        genres=np.array(final_genres),
        filter_indices=np.array(final_indices)
    )
    print(f"  Saved to: {output_path}")
    
    # --- Summary ---
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Original:        {len(piece_ids)} pieces")
    print(f"Pass quality:    {len(piece_ids) - quality_fail}")
    print(f"Have genre:      {len(piece_ids) - quality_fail - no_genre}")
    print(f"Not classical:   {len(filtered_data)}")
    print(f"Final sample:    {len(final_data)}")
    print(f"\nGenre distribution:")
    for genre, count in Counter(final_genres).most_common():
        print(f"  {genre}: {count}")
    
    return final_dists, final_pids, final_genres


if __name__ == "__main__":
    load_and_filter()
