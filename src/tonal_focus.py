#!/usr/bin/env python3
"""
Tonal Focus — measuring how tightly pitch content clusters near the tonic.

This script implements the tonal focus measure described in §3.4.1 of:
    "Two Routes to Tonal Coherence" (Xu & Hall, SMC 2026)

Definition (Eq. 1):
    Tonal Focus_k = Σ_{i=c-k}^{c+k} d_i

where d is a normalized 35-D pitch-class distribution on the line-of-fifths,
c is the tonal center index, and k is the window half-width.

Higher values (→1.0) indicate tight diatonic concentration near the tonic.
Lower values indicate chromatic extension away from the central region.

Usage:
    # As a library
    from src.tonal_focus import compute_tonal_focus
    focus = compute_tonal_focus(distribution, tonal_center=17, k=3)

    # From command line — compute focus for all pieces in a corpus
    python -m src.tonal_focus results/data/classical_pitch_class_distributions.npz

    # With custom k threshold
    python -m src.tonal_focus results/data/lmd_filtered_with_genre.npz --k 5
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

LOF_SIZE = 35          # Line-of-fifths dimension (F♭♭ to A♯♯)
CENTER_IDX = 17        # C natural = index 17


def compute_tonal_focus(distribution, tonal_center, k=3):
    """
    Compute tonal focus: proportion of pitch content within ±k of the tonal center
    on the line-of-fifths.

    Parameters
    ----------
    distribution : array-like, shape (35,) or (12,)
        Normalized pitch-class distribution. Need not sum to 1 (will be normalized).
    tonal_center : int
        Index of the tonal center on the line-of-fifths (0–34), or chromatic (0–11).
    k : int, default=3
        Window half-width. k=3 captures a diatonic-scale-sized neighbourhood.

    Returns
    -------
    float
        Tonal focus value in [0, 1]. Higher = more concentrated near tonic.

    Examples
    --------
    >>> import numpy as np
    >>> d = np.zeros(35); d[16:19] = [0.2, 0.5, 0.3]  # F, C, G
    >>> compute_tonal_focus(d, tonal_center=17, k=3)
    1.0
    """
    d = np.asarray(distribution, dtype=float)
    if d.sum() == 0:
        return np.nan

    # Normalize
    d = d / d.sum()

    c = int(tonal_center)
    dim = len(d)

    # Compute focus using distance mask (handles boundary correctly)
    indices = np.arange(dim)
    if dim == 12:
        # Circular distance for 12-D chroma
        dists = np.minimum(np.abs(indices - c), dim - np.abs(indices - c))
    else:
        # Linear distance for 35-D line-of-fifths
        dists = np.abs(indices - c)

    mask = dists <= k
    return float(d[mask].sum())


def compute_tonal_focus_batch(distributions, tonal_centers, k=3):
    """
    Compute tonal focus for a batch of pieces.

    Parameters
    ----------
    distributions : array, shape (N, D)
        Pitch-class distributions (D = 35 or 12).
    tonal_centers : array-like of int, shape (N,)
        Tonal center index for each piece.
    k : int
        Window half-width.

    Returns
    -------
    np.ndarray, shape (N,)
        Tonal focus values.
    """
    return np.array([
        compute_tonal_focus(d, c, k=k)
        for d, c in zip(distributions, tonal_centers)
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute tonal focus for a corpus of pitch-class distributions")
    parser.add_argument('input', type=str,
                        help="Path to .npz file with 'distributions' and 'piece_ids'")
    parser.add_argument('--k', type=int, default=3,
                        help="Window half-width (default: 3)")
    parser.add_argument('--output', type=str, default=None,
                        help="Output CSV path (default: print to stdout)")
    args = parser.parse_args()

    # Load data
    data = np.load(args.input, allow_pickle=True)
    distributions = data['distributions']
    piece_ids = data['piece_ids']
    print(f"Loaded {len(distributions)} pieces from {args.input}")
    print(f"Distribution dimensions: {distributions.shape[1]}-D")

    # Estimate tonal centers (argmax as simple heuristic)
    centers = np.argmax(distributions, axis=1)
    print(f"Using argmax tonal centers (for K-S estimation, use src.utils.loaders)")

    # Compute focus
    focus_vals = compute_tonal_focus_batch(distributions, centers, k=args.k)
    valid = ~np.isnan(focus_vals)

    print(f"\nTonal Focus (k={args.k}):")
    print(f"  N = {valid.sum()}")
    print(f"  Mean = {np.nanmean(focus_vals):.4f}")
    print(f"  Median = {np.nanmedian(focus_vals):.4f}")
    print(f"  SD = {np.nanstd(focus_vals):.4f}")
    print(f"  Range = [{np.nanmin(focus_vals):.4f}, {np.nanmax(focus_vals):.4f}]")

    if args.output:
        import pandas as pd
        df = pd.DataFrame({
            'piece_id': piece_ids,
            'tonal_center': centers,
            f'focus_k{args.k}': focus_vals,
        })
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
