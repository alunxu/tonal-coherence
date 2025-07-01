#!/usr/bin/env python3
"""
Unified table / robustness statistics generator (Appendix B).

Generates all results tables for the paper appendix:
  B.1  Non-parametric effect sizes
  B.2  Modulation stratification
  B.3  Key estimation validation
  B.4  12-D reanalysis + windowed analysis
  B.5  Correlation matrices

Usage:
  python -m src.generate_tables          # run all
  python -m src.generate_tables --table 1 3
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from scipy.stats import entropy, kurtosis, kruskal, ttest_rel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.loaders import (
    RESULTS_DIR, get_annotated_tonal_centers, get_ks_tonal_centers,
    load_all_tdm_results,
)
from src.utils.stats import (
    cohens_d, cliffs_delta, bootstrap_ci_d, rank_biserial,
    compute_kurtosis_from_weights,
    compute_entropy_from_weights, compute_kurtosis_lof,
)
from src.tonal_focus import compute_tonal_focus

TDM_DIR = RESULTS_DIR / "tdm_analysis" / "improved_tdm_results"
DATA_DIR = RESULTS_DIR / "data"


# =========================================================================
# Shared data loading
# =========================================================================

def _load_tdm_results():
    """Load main TDM results CSV."""
    return load_all_tdm_results()


def _load_focus_data():
    """Load distributions and compute tonal focus for both corpora."""
    c_pkg = np.load(DATA_DIR / "classical_pitch_class_distributions.npz",
                    allow_pickle=True)
    p_pkg = np.load(DATA_DIR / "lmd_filtered_with_genre.npz",
                    allow_pickle=True)
    rows = []
    # Classical
    centers = get_annotated_tonal_centers(c_pkg['piece_ids'])
    for i, ctr in enumerate(centers):
        if ctr is not None:
            f = compute_tonal_focus(c_pkg['distributions'][i], ctr, k=3)
            rows.append({'piece_id': Path(c_pkg['piece_ids'][i]).name,
                         'focus': f, 'corpus': 'Classical'})
    # Pop
    ks_centers = get_ks_tonal_centers(p_pkg['distributions'])
    for i, ctr in enumerate(ks_centers):
        if ctr is not None:
            f = compute_tonal_focus(p_pkg['distributions'][i], ctr, k=3)
            rows.append({'piece_id': str(p_pkg['piece_ids'][i]),
                         'focus': f, 'corpus': 'Pop'})
    return pd.DataFrame(rows)


# =========================================================================
# Table B.1 — Non-parametric effect sizes (§B.1)
# =========================================================================

def table_effect_sizes():
    """Compute Cohen's d, Cliff's delta, rank-biserial for all metrics."""
    print("=" * 70)
    print("TABLE B.1: NON-PARAMETRIC EFFECT SIZES")
    print("=" * 70)

    df = _load_tdm_results()
    df = df[df['converged'] == True]
    df['fifth_dom'] = df['+P5'] + df['-P5']
    df['kurtosis'] = df.apply(compute_kurtosis_from_weights, axis=1)
    df['entropy'] = df.apply(compute_entropy_from_weights, axis=1)

    c = df[df['corpus'] == 'Classical']
    p = df[df['corpus'] == 'Pop']

    focus_df = _load_focus_data()

    metrics = [
        ('lambda', c['lambda'], p['lambda']),
        ('Fifth Dominance', c['fifth_dom'], p['fifth_dom']),
        ('Weight Entropy', c['entropy'], p['entropy']),
        ('Weight Kurtosis', c['kurtosis'], p['kurtosis']),
    ]
    c_foc = focus_df[focus_df['corpus'] == 'Classical']['focus'].dropna()
    p_foc = focus_df[focus_df['corpus'] == 'Pop']['focus'].dropna()
    if len(c_foc) > 0 and len(p_foc) > 0:
        metrics.append(('Tonal Focus (k=3)', c_foc, p_foc))

    print(f"\n{'Metric':<20} {'d':>8} {'[95% CI]':>18} {'Cliff δ':>10} "
          f"{'r_rb':>8} {'Cl Mean':>10} {'Pop Mean':>10}")
    print("-" * 90)
    for name, g1, g2 in metrics:
        g1v, g2v = g1.dropna().values, g2.dropna().values
        d = cohens_d(g1v, g2v)
        lo, hi = bootstrap_ci_d(g1v, g2v)
        cliff = cliffs_delta(g1v, g2v)
        rb = rank_biserial(g1v, g2v)
        print(f"{name:<20} {d:>8.3f} [{lo:>7.3f}, {hi:>6.3f}] {cliff:>10.3f} "
              f"{rb:>8.3f} {np.mean(g1v):>10.3f} {np.mean(g2v):>10.3f}")


# =========================================================================
# Table B.2 — Modulation stratification (§B.2)
# =========================================================================

def table_modulation():
    """Stratify classical corpus by modulation rate and test stability."""
    print("\n" + "=" * 70)
    print("TABLE B.2: MODULATION STRATIFICATION")
    print("=" * 70)

    tdm_path = TDM_DIR / "classical_35d_tdm_corrected.csv"
    mod_path = DATA_DIR / "classical_modulation_rates.csv"

    try:
        df_tdm = pd.read_csv(tdm_path)
        df_mod = pd.read_csv(mod_path)
    except FileNotFoundError as e:
        print(f"Skipping: {e}")
        return

    df = pd.merge(df_tdm, df_mod, on='piece_id', how='inner')
    print(f"Merged: {len(df)} pieces")

    q = df['modulation_rate'].quantile([0.333, 0.667])
    df['stratum'] = pd.cut(df['modulation_rate'],
                           bins=[-1, q[0.333], q[0.667], 999],
                           labels=['Low', 'Medium', 'High'])

    intervals = ['+P5', '-P5', '+M3', '-M3', '+m3', '-m3']
    print(f"\n{'Stratum':<10} {'N':>5} {'Kurtosis':>16} {'5th Dom':>16} "
          f"{'Lambda':>16}")
    print("-" * 70)
    for s in ['Low', 'Medium', 'High']:
        sub = df[df['stratum'] == s]
        w = sub[intervals].values
        w = w / (w.sum(axis=1, keepdims=True) + 1e-10)
        k = kurtosis(w, axis=1)
        fd = w[:, 0] + w[:, 1]
        lam = sub['lambda']
        print(f"{s:<10} {len(sub):>5} {np.mean(k):>7.2f} ± {np.std(k):<6.2f} "
              f"{np.mean(fd):>7.2f} ± {np.std(fd):<6.2f} "
              f"{lam.mean():>7.2f} ± {lam.std():<6.2f}")
        df.loc[df['stratum'] == s, 'Kurtosis'] = k
        df.loc[df['stratum'] == s, 'FifthDom'] = fd

    print("\nStability (Kruskal-Wallis p):")
    groups = [df[df['stratum'] == s] for s in ['Low', 'Medium', 'High']]
    for m in ['lambda', 'FifthDom', 'Kurtosis']:
        _, p = kruskal(*[g[m].values for g in groups])
        print(f"  {m}: p = {p:.3e} {'(Stable)' if p >= 0.05 else '(Sig.)'}")


# =========================================================================
# Table B.3 — Key estimation validation (§B.3)
# =========================================================================

def table_key_estimation():
    """Validate K-S key estimation against argmax and expert annotations."""
    print("\n" + "=" * 70)
    print("TABLE B.3: KEY ESTIMATION VALIDATION")
    print("=" * 70)

    # Classical validation
    try:
        path_c = DATA_DIR / "classical_pitch_class_distributions.npz"
        data_c = np.load(path_c, allow_pickle=True)
        dists_c, ids_c = data_c['distributions'], data_c['piece_ids']
        annotated = get_annotated_tonal_centers(ids_c)
        mask = [k is not None for k in annotated]
        dists_c = dists_c[mask]
        ann_keys = np.array([k for k in annotated if k is not None])
        argmax_c = np.argmax(dists_c, axis=1)
        diffs = argmax_c - ann_keys
        exact = np.sum(diffs == 0)
        near = np.sum(np.abs(diffs) <= 1)
        n = len(dists_c)
        print(f"\nClassical (argmax vs expert) N={n}:")
        print(f"  Exact: {exact / n * 100:.1f}%")
        print(f"  ±1 fifth: {near / n * 100:.1f}%")
    except Exception as e:
        print(f"Classical check failed: {e}")

    # Pop validation
    try:
        path_p = DATA_DIR / "lmd_filtered_with_genre.npz"
        if not path_p.exists():
            path_p = DATA_DIR / "lmd_pitch_class_distributions_35d_partitura.npz"
        data_p = np.load(path_p, allow_pickle=True)
        dists = data_p['distributions']
        argmax_keys = np.argmax(dists, axis=1)
        ks_keys = get_ks_tonal_centers(dists)

        if dists.shape[1] == 35:
            argmax_chroma = ((argmax_keys - 17) * 7) % 12
        else:
            argmax_chroma = argmax_keys

        diffs = (argmax_chroma - ks_keys) % 12
        n = len(dists)
        exact = np.sum(diffs == 0)
        dom = np.sum(diffs == 7)
        sub = np.sum(diffs == 5)
        print(f"\nPop (argmax vs K-S) N={n}:")
        print(f"  Exact: {exact / n * 100:.1f}%")
        print(f"  Dominant (V): {dom / n * 100:.1f}%")
        print(f"  Subdominant (IV): {sub / n * 100:.1f}%")
        print(f"  Total ±1 fifth: {(exact + dom + sub) / n * 100:.1f}%")
    except Exception as e:
        print(f"Pop check failed: {e}")


# =========================================================================
# Table B.4 — 12-D robustness + windowed analysis (§B.4)
# =========================================================================

def table_robustness_12d():
    """Compare 35-D vs 12-D effect sizes."""
    print("\n" + "=" * 70)
    print("TABLE B.4 (top): 12-D ROBUSTNESS")
    print("=" * 70)

    try:
        df_c = pd.read_csv(TDM_DIR / "classical_12d_tdm.csv")
        df_p = pd.read_csv(TDM_DIR / "pop_12d_tdm.csv")
    except FileNotFoundError as e:
        print(f"Skipping: {e}")
        return

    # Filter pop to clean set
    try:
        pkg = np.load(DATA_DIR / "lmd_filtered_with_genre.npz",
                      allow_pickle=True)
        df_p['md5'] = df_p['piece_id'].astype(str).apply(
            lambda x: Path(x).stem)
        target = set(t.replace('.mid', '') for t in pkg['piece_ids'])
        df_p = df_p[df_p['md5'].isin(target)]
    except Exception:
        pass

    for df in [df_c, df_p]:
        df['fifth_dom'] = df['+P5'] + df['-P5']
        df['kurtosis'] = df.apply(compute_kurtosis_lof, axis=1)

    print(f"\n{'Metric':<15} {'Cl Mean':>10} {'Pop Mean':>10} {'d':>8}")
    print("-" * 50)
    for m in ['lambda', 'kurtosis', 'fifth_dom']:
        g1 = df_c[m].dropna().values
        g2 = df_p[m].dropna().values
        d = cohens_d(g1, g2)
        print(f"{m:<15} {np.mean(g1):>10.3f} {np.mean(g2):>10.3f} {d:>8.3f}")


def table_windowed():
    """Compare global vs windowed (opening 16 bars) TDM metrics."""
    print("\n" + "=" * 70)
    print("TABLE B.4 (bottom): WINDOWED ANALYSIS")
    print("=" * 70)

    from src.models.tonal_diffusion import ImprovedTDM

    global_res = _load_tdm_results()

    try:
        win_c = np.load(DATA_DIR / "classical_windowed_16bar.npz",
                        allow_pickle=True)
        win_p = np.load(DATA_DIR / "pop_windowed_16bar_35d.npz",
                        allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Skipping: {e}")
        return

    tdm = ImprovedTDM(dims=35)
    N_SUB = 300

    for label, ids, dists in [('Classical', win_c['piece_ids'],
                                win_c['distributions']),
                               ('Pop', win_p['piece_ids'],
                                win_p['distributions'])]:
        if len(ids) > N_SUB:
            idx = np.random.choice(len(ids), N_SUB, replace=False)
            ids, dists = ids[idx], dists[idx]

        from tqdm import tqdm
        win_lam = []
        win_ids = []
        for pid, dist in tqdm(zip(ids, dists), total=len(ids),
                               desc=f"TDM {label}"):
            if dist.sum() == 0:
                continue
            res = tdm.infer_multistart(dist, n_starts=5)
            win_lam.append(res['lambda'])
            win_ids.append(pid)

        df_win = pd.DataFrame({'piece_id': win_ids, 'win_lambda': win_lam})
        g = global_res[global_res['corpus'] == label].copy()
        if label == 'Pop':
            g['piece_id'] = g['piece_id'].astype(str).apply(
                lambda x: Path(x).stem)
        g = g[['piece_id', 'lambda']].rename(columns={'lambda': 'global_lam'})
        merged = pd.merge(df_win, g, on='piece_id')
        gl, wl = merged['global_lam'], merged['win_lambda']
        pct = (wl.mean() - gl.mean()) / gl.mean() * 100
        print(f"\n{label} (N={len(merged)}):")
        print(f"  Global λ: {gl.mean():.3f} ± {gl.std():.3f}")
        print(f"  Window λ: {wl.mean():.3f} ± {wl.std():.3f}")
        print(f"  Change: {pct:+.1f}%")
        if len(merged) > 1:
            _, p = ttest_rel(gl, wl)
            print(f"  Paired t: p = {p:.3e}")


# =========================================================================
# Table B.5 — Correlation matrices (§B.5)
# =========================================================================

def table_correlations():
    """Compute pairwise correlation matrices for both corpora."""
    print("\n" + "=" * 70)
    print("TABLE B.5: CORRELATION MATRICES")
    print("=" * 70)

    df = _load_tdm_results()
    df = df[df['converged'] == True]
    df['fifth_dom'] = df['+P5'] + df['-P5']
    df['kurtosis'] = df.apply(compute_kurtosis_from_weights, axis=1)
    df['entropy'] = df.apply(compute_entropy_from_weights, axis=1)

    cols = ['lambda', 'fifth_dom', 'entropy', 'kurtosis']
    for corpus in ['Classical', 'Pop']:
        sub = df[df['corpus'] == corpus][cols].dropna()
        corr = sub.corr()
        print(f"\n{corpus} (N={len(sub)}):")
        print(corr.round(2).to_string())


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all appendix table statistics")
    parser.add_argument('--table', nargs='*', type=int,
                        help="Table numbers to generate (1-5). Default: all")
    args = parser.parse_args()

    tables = {
        1: ('B.1: Effect sizes', table_effect_sizes),
        2: ('B.2: Modulation stratification', table_modulation),
        3: ('B.3: Key estimation', table_key_estimation),
        4: ('B.4: 12-D + windowed', lambda: (table_robustness_12d(),
                                              table_windowed())),
        5: ('B.5: Correlations', table_correlations),
    }

    selected = args.table if args.table else list(tables.keys())
    for n in selected:
        if n in tables:
            name, fn = tables[n]
            print(f"\n{'#' * 70}")
            print(f"# {name}")
            print(f"{'#' * 70}")
            fn()
