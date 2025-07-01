#!/usr/bin/env python3
"""
Tonal Connection — measuring structured intervallic exploration of tonal space.

This script implements the tonal connection measure described in §3.4.2 of:
    "Two Routes to Tonal Coherence" (Xu & Hall, SMC 2026)

The Tonal Diffusion Model (TDM) [Lieck et al. 2020] models pitch-class
distributions as arising from random walks on the Tonnetz. Given a piece's
pitch-class distribution d and tonal center c:

    {λ*, w*} = argmax_{λ,w} Σ_i d_i log P_model(i | λ, w, c)

Tonal connection is the fitted path-length parameter:

    Tonal Connection = λ*

Higher λ means more extensive traversal of tonal space while maintaining
structured intervallic connections back to the tonic.

The fitted weight vector w* = [w_{-4}, w_{-3}, w_{-1}, w_{+1}, w_{+3}, w_{+4}]
reveals HOW the piece navigates tonal space. Three summary statistics
characterize this:

    Fifth dominance  = (w_{-1} + w_{+1}) / Σ w_j
    Weight entropy   = -Σ w_j log w_j
    Weight kurtosis  = Fisher's excess kurtosis of w

Usage:
    # As a library
    from src.tonal_connection import compute_tonal_connection
    result = compute_tonal_connection(distribution, tonal_center=17)
    print(result['lambda'], result['fifth_dominance'])

    # From command line — compute for all pieces in a corpus
    python -m src.tonal_connection results/data/classical_pitch_class_distributions.npz

    # With specific tonal center source
    python -m src.tonal_connection data.npz --centers annotated
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis as sp_kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.tonal_diffusion import ImprovedTDM


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

INTERVAL_NAMES = ['+P5', '-P5', '+M3', '-M3', '+m3', '-m3']


def compute_tonal_connection(distribution, tonal_center, dims=35, n_starts=5):
    """
    Compute tonal connection (λ) and interval weight structure for one piece.

    Parameters
    ----------
    distribution : array-like, shape (dims,)
        Normalized pitch-class distribution (35-D line-of-fifths or 12-D chroma).
    tonal_center : int
        Index of the tonal center.
    dims : int
        Dimensionality (35 or 12).
    n_starts : int
        Number of random starts for MLE optimization.

    Returns
    -------
    dict or None
        Keys: 'lambda', 'weights' (6-element array), 'tonal_center', 'converged',
              'fifth_dominance', 'weight_entropy', 'weight_kurtosis'.
        Returns None if fitting fails (e.g., degenerate distribution).

    Examples
    --------
    >>> import numpy as np
    >>> d = np.zeros(35); d[16:19] = [0.2, 0.5, 0.3]
    >>> r = compute_tonal_connection(d, tonal_center=17)
    >>> print(f"λ = {r['lambda']:.2f}, fifth dom = {r['fifth_dominance']:.2f}")
    """
    d = np.asarray(distribution, dtype=float)
    if d.sum() == 0:
        return None

    model = ImprovedTDM(dims=dims)
    result = model.infer_multistart(d, n_starts=n_starts,
                                     forced_center=int(tonal_center))
    if result is None:
        return None

    w = result['weights']

    # Fifth dominance: proportion of weight on ±P5
    fifth_dom = (w[0] + w[1]) / w.sum() if w.sum() > 0 else np.nan

    # Weight entropy (nats → bits)
    w_safe = w / (w.sum() + 1e-10)
    w_safe = w_safe[w_safe > 0]
    w_entropy = -np.sum(w_safe * np.log2(w_safe))

    # Weight kurtosis (Fisher's excess)
    w_kurt = sp_kurtosis(w, fisher=True)

    return {
        'lambda': result['lambda'],
        'weights': w,
        'tonal_center': result['tonal_center'],
        'converged': result['converged'],
        'fifth_dominance': fifth_dom,
        'weight_entropy': w_entropy,
        'weight_kurtosis': w_kurt,
    }


def compute_tonal_connection_batch(distributions, tonal_centers, dims=35,
                                    n_starts=3, verbose=True):
    """
    Compute tonal connection for a batch of pieces.

    Parameters
    ----------
    distributions : array, shape (N, D)
    tonal_centers : array-like of int, shape (N,)
    dims : int
    n_starts : int
    verbose : bool
        If True, show progress bar.

    Returns
    -------
    list[dict]
        One result dict per piece (None entries for failed fits).
    """
    results = []
    iterator = zip(distributions, tonal_centers)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(list(iterator), desc="Fitting TDM")

    for d, c in iterator:
        if c is None or d.sum() == 0:
            results.append(None)
        else:
            results.append(compute_tonal_connection(d, c, dims=dims,
                                                     n_starts=n_starts))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute tonal connection (TDM λ) for a corpus")
    parser.add_argument('input', type=str,
                        help="Path to .npz file with 'distributions' and 'piece_ids'")
    parser.add_argument('--centers', type=str, default='argmax',
                        choices=['argmax', 'annotated', 'ks'],
                        help="Tonal center estimation method (default: argmax)")
    parser.add_argument('--n-starts', type=int, default=3,
                        help="Number of MLE random starts (default: 3)")
    parser.add_argument('--max-pieces', type=int, default=None,
                        help="Max pieces to process (for testing)")
    parser.add_argument('--output', type=str, default=None,
                        help="Output CSV path (default: print summary)")
    args = parser.parse_args()

    # Load data
    data = np.load(args.input, allow_pickle=True)
    distributions = data['distributions']
    piece_ids = data['piece_ids']
    dims = distributions.shape[1]
    print(f"Loaded {len(distributions)} pieces ({dims}-D) from {args.input}")

    # Subsample if requested
    if args.max_pieces and len(distributions) > args.max_pieces:
        idx = np.random.choice(len(distributions), args.max_pieces, replace=False)
        distributions = distributions[idx]
        piece_ids = piece_ids[idx]
        print(f"Subsampled to {len(distributions)} pieces")

    # Tonal centers
    if args.centers == 'argmax':
        centers = np.argmax(distributions, axis=1)
        print("Using argmax tonal centers")
    elif args.centers == 'ks':
        from src.utils.loaders import get_ks_tonal_centers
        centers = get_ks_tonal_centers(distributions)
        print("Using Krumhansl-Schmuckler tonal centers")
    elif args.centers == 'annotated':
        from src.utils.loaders import get_annotated_tonal_centers
        centers = get_annotated_tonal_centers(piece_ids)
        print("Using annotated tonal centers")

    # Compute
    results = compute_tonal_connection_batch(
        distributions, centers, dims=dims, n_starts=args.n_starts)

    # Summary
    lambdas = [r['lambda'] for r in results if r is not None]
    fifths = [r['fifth_dominance'] for r in results if r is not None]
    kurts = [r['weight_kurtosis'] for r in results if r is not None]

    print(f"\nTonal Connection (λ):")
    print(f"  N = {len(lambdas)}")
    print(f"  Mean = {np.mean(lambdas):.4f}")
    print(f"  Median = {np.median(lambdas):.4f}")
    print(f"  SD = {np.std(lambdas):.4f}")

    print(f"\nFifth Dominance:")
    print(f"  Mean = {np.mean(fifths):.4f}")

    print(f"\nWeight Kurtosis:")
    print(f"  Mean = {np.mean(kurts):.4f}")

    if args.output:
        import pandas as pd
        rows = []
        for pid, r in zip(piece_ids, results):
            if r is not None:
                row = {'piece_id': pid, 'lambda': r['lambda'],
                       'tonal_center': r['tonal_center'],
                       'fifth_dominance': r['fifth_dominance'],
                       'weight_entropy': r['weight_entropy'],
                       'weight_kurtosis': r['weight_kurtosis'],
                       'converged': r['converged']}
                for i, name in enumerate(INTERVAL_NAMES):
                    row[name] = r['weights'][i]
                rows.append(row)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nSaved {len(rows)} results to {args.output}")


if __name__ == "__main__":
    main()
