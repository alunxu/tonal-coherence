#!/usr/bin/env python3
"""
run_tdm_filtered.py

Run ImprovedTDM analysis on:
1. Classical Corpus (using annotated tonal centers)
2. Filtered LMD Corpus (using K-S estimated tonal centers)

Output: CSV files with lambda, weights, and metadata.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mannwhitneyu

from src.models.tonal_diffusion import ImprovedTDM
from src.utils.loaders import (
    RESULTS_DIR, TDM_RESULTS_DIR,
    load_classical_35d_distributions,
    get_annotated_tonal_centers,
    get_ks_tonal_centers
)

def run_analysis():
    print('='*70)
    print('TDM ANALYSIS: CORRECTED POISSON + FILTERED DATA')
    print('='*70)

    # --- Load Classical ---
    print('\nLoading Classical...')
    classical = load_classical_35d_distributions()
    c_dists = classical['distributions']
    c_ids = classical['piece_ids'].tolist()
    c_centers = get_annotated_tonal_centers(c_ids)
    
    # Filter to pieces with annotations
    valid_c_idx = [i for i, c in enumerate(c_centers) if c is not None]
    c_dists = c_dists[valid_c_idx]
    c_ids = [c_ids[i] for i in valid_c_idx]
    c_centers = [c_centers[i] for i in valid_c_idx]
    
    print(f'  {len(c_ids)} pieces with annotated tonal centers')

    # --- Load Filtered Pop ---
    print('\nLoading Filtered Pop...')
    lmd_path = RESULTS_DIR / 'data' / 'lmd_filtered_with_genre.npz'
    pop_data = np.load(lmd_path, allow_pickle=True)
    p_dists = pop_data['distributions']
    p_ids = pop_data['piece_ids'].tolist()
    p_genres = pop_data['genres'].tolist()
    
    print('  Estimating keys with K-S...')
    p_centers = get_ks_tonal_centers(p_dists)
    print(f'  {len(p_ids)} pieces (balanced & genre-filtered)')

    # --- Initialize Model ---
    model = ImprovedTDM(dims=35)

    # --- Analyze Classical ---
    print('\nAnalyzing Classical...')
    c_results = []
    for i, (dist, pid, center) in enumerate(tqdm(zip(c_dists, c_ids, c_centers), total=len(c_ids))):
        if dist.sum() == 0:
            continue
        
        result = model.infer_multistart(dist, n_starts=3, forced_center=center)
        if result:
            row = {
                'piece_id': pid,
                'lambda': result['lambda'],
                'tonal_center': result['tonal_center'],
                'converged': result['converged'],
                'corpus': 'Classical',
                'genre': 'Classical'
            }
            # Add weights
            for j, name in enumerate(model.interval_names):
                row[name] = result['weights'][j]
            c_results.append(row)

    df_c = pd.DataFrame(c_results)
    print(f'  λ = {df_c["lambda"].mean():.3f} ± {df_c["lambda"].std():.3f}')

    # --- Analyze Pop ---
    print('\nAnalyzing Pop...')
    p_results = []
    for i, (dist, pid, center, genre) in enumerate(tqdm(zip(p_dists, p_ids, p_centers, p_genres), total=len(p_ids))):
        if center is None or dist.sum() == 0:
            continue
            
        result = model.infer_multistart(dist, n_starts=3, forced_center=center)
        if result:
            row = {
                'piece_id': pid,
                'lambda': result['lambda'],
                'tonal_center': result['tonal_center'],
                'converged': result['converged'],
                'corpus': 'Pop',
                'genre': genre
            }
            # Add weights
            for j, name in enumerate(model.interval_names):
                row[name] = result['weights'][j]
            p_results.append(row)

    df_p = pd.DataFrame(p_results)
    print(f'  λ = {df_p["lambda"].mean():.3f} ± {df_p["lambda"].std():.3f}')

    # --- Compare ---
    print('\n' + '='*70)
    print('COMPARISON')
    print('='*70)
    print(f'Classical: λ = {df_c["lambda"].mean():.3f} ± {df_c["lambda"].std():.3f} (n={len(df_c)})')
    print(f'Pop:       λ = {df_p["lambda"].mean():.3f} ± {df_p["lambda"].std():.3f} (n={len(df_p)})')

    stat, p_val = mannwhitneyu(df_c['lambda'], df_p['lambda'])
    d = (df_c['lambda'].mean() - df_p['lambda'].mean()) / np.sqrt((df_c['lambda'].std()**2 + df_p['lambda'].std()**2) / 2)
    print(f'Mann-Whitney p = {p_val:.2e}')
    print(f"Cohen's d = {d:.3f} (Classical - Pop)")

    # --- Save ---
    output_dir = TDM_RESULTS_DIR / 'improved_tdm_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_c.to_csv(output_dir / 'classical_35d_tdm_corrected.csv', index=False)
    # df_p has genre column now!
    df_p.to_csv(output_dir / 'pop_35d_filtered_corrected.csv', index=False)
    print(f'\nSaved to {output_dir}')
    
    # Save combined
    df_all = pd.concat([df_c, df_p], ignore_index=True)
    df_all.to_csv(output_dir / 'all_tdm_results.csv', index=False)

if __name__ == "__main__":
    run_analysis()
