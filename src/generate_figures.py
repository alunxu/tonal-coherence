#!/usr/bin/env python3
"""
Unified figure generator — all paper figures ordered by narrative flow.

  Figure 1  — Conceptual overview (hand-made, not generated)
  Figure 2  — Main results: focus/connection violins + scatter (§4.1)
  Figure 3  — Dimension characterization: weights + ridge (§4.3)
  Figure 4  — Historical trajectory (§4.4)
  Figure 5  — Genre breakdown (App A)
  Figure 6  — Composer boxplots (App A)

Usage:
  python -m src.generate_figures            # generate all
  python -m src.generate_figures --fig 2 4  # selected figures
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch, Ellipse
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.loaders import (
    RESULTS_DIR, FIGURES_DIR, TDM_RESULTS_DIR,
    get_annotated_tonal_centers, get_ks_tonal_centers,
    load_all_tdm_results,
)
from src.utils.stats import (
    cohens_d, cliffs_delta, bootstrap_ci_d,
    compute_kurtosis_from_weights,
    sig_stars, draw_bracket,
)
from src.tonal_focus import compute_tonal_focus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {'classical': '#2E5090', 'pop': '#E85D3D'}
C_COLOR, P_COLOR = COLORS['classical'], COLORS['pop']


# =========================================================================
# Style
# =========================================================================

def _set_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })


def _save(fig, name):
    """Save figure to figures/."""
    for ext in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=300,
                    bbox_inches='tight')
    print(f"  ✅ Saved {name} → {FIGURES_DIR}")
    plt.close(fig)


# =========================================================================
# Shared data loading for Figures 2, 3
# =========================================================================

def _load_main_data():
    """Load TDM results + distributions, return matched arrays."""
    from scipy.stats import entropy as sp_entropy

    df = load_all_tdm_results()
    df = df[df['converged'] == True]
    df = df[(df['lambda'] > 0.05) & (df['lambda'] < 4.95)]

    c_pkg = np.load(RESULTS_DIR / "data/classical_pitch_class_distributions.npz",
                    allow_pickle=True)
    p_pkg = np.load(RESULTS_DIR / "data/lmd_filtered_with_genre.npz",
                    allow_pickle=True)

    c_dist_map = dict(zip(c_pkg['piece_ids'], c_pkg['distributions']))
    p_dist_map = {}
    for pid, dist in zip(p_pkg['piece_ids'], p_pkg['distributions']):
        p_dist_map[pid] = dist
        p_dist_map[str(pid).replace('.mid', '')] = dist

    filtered_ids = set()
    for pid in p_pkg['piece_ids']:
        filtered_ids.add(str(pid))
        filtered_ids.add(str(pid).replace('.mid', ''))

    c_lambda, c_focus, p_lambda, p_focus = [], [], [], []
    c_rows, p_rows = [], []

    for _, row in df.iterrows():
        pid, corpus = row['piece_id'], row['corpus']
        lam, center = row['lambda'], row['tonal_center']
        if pd.isna(lam) or pd.isna(center):
            continue
        if corpus == 'Pop' and str(pid) not in filtered_ids:
            continue

        dist = (c_dist_map.get(pid) if corpus == 'Classical'
                else p_dist_map.get(pid))
        if dist is None and corpus == 'Pop':
            dist = p_dist_map.get(str(pid).replace('.mid', ''))
        if dist is None:
            continue

        dist_norm = dist / (dist.sum() + 1e-10)
        c_idx = int(center)
        mask = np.abs(np.arange(len(dist_norm)) - c_idx) <= 3
        f = dist_norm[mask].sum()

        if corpus == 'Pop':
            ent = sp_entropy(dist_norm + 1e-10, base=2)
            if ent > 3.2 or f < 0.3:
                continue

        if corpus == 'Classical':
            c_lambda.append(lam); c_focus.append(f); c_rows.append(row)
        else:
            p_lambda.append(lam); p_focus.append(f); p_rows.append(row)

    print(f"  Loaded {len(c_lambda)} classical, {len(p_lambda)} pop pieces")
    return (np.array(c_lambda), np.array(c_focus), pd.DataFrame(c_rows),
            np.array(p_lambda), np.array(p_focus), pd.DataFrame(p_rows))


# =========================================================================
# Figure 2 — Main results (§4.1)
# =========================================================================

def figure2():
    """Three-panel figure: focus violin, lambda violin, scatter with archetypes."""
    _set_style()
    print("Generating Figure 2 (main results)...")

    c_lam, c_foc, _, p_lam, p_foc, _ = _load_main_data()

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # (a) Tonal Focus
    ax = axes[0]
    parts = ax.violinplot([c_foc, p_foc], positions=[1, 2],
                          showextrema=False, widths=0.7)
    parts['bodies'][0].set_facecolor(C_COLOR); parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_facecolor(P_COLOR); parts['bodies'][1].set_alpha(0.7)
    ax.boxplot([c_foc, p_foc], positions=[1, 2], widths=0.15,
               patch_artist=True, boxprops=dict(facecolor='white', alpha=0.5),
               medianprops=dict(color='black', linewidth=2), showfliers=False)
    ax.set_ylabel('Tonal Focus (k = 3)', fontsize=28, fontweight='bold')
    ax.set_title('(a) Tonal Focus', fontweight='bold', fontsize=32)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis='y', labelsize=26)
    c_med, p_med = np.median(c_foc), np.median(p_foc)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Classical\n(Mdn = {c_med:.2f})',
                        f'Pop\n(Mdn = {p_med:.2f})'],
                       fontsize=24, fontweight='bold')
    d = cohens_d(c_foc, p_foc)
    draw_bracket(ax, 1, 2, 1.04, f'd = {d:.2f}{sig_stars(d)}')

    # (b) Tonal Connection
    ax = axes[1]
    parts = ax.violinplot([c_lam, p_lam], positions=[1, 2],
                          showextrema=False, widths=0.7)
    parts['bodies'][0].set_facecolor(C_COLOR); parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_facecolor(P_COLOR); parts['bodies'][1].set_alpha(0.7)
    ax.boxplot([c_lam, p_lam], positions=[1, 2], widths=0.15,
               patch_artist=True, boxprops=dict(facecolor='white', alpha=0.5),
               medianprops=dict(color='black', linewidth=2), showfliers=False)
    ax.set_ylabel('Tonal Connection', fontsize=28, fontweight='bold')
    ax.set_title('(b) Tonal Connection', fontweight='bold', fontsize=32)
    ax.set_ylim(0, 5.5)
    ax.tick_params(axis='y', labelsize=26)
    c_med, p_med = np.median(c_lam), np.median(p_lam)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Classical\n(Mdn = {c_med:.2f})',
                        f'Pop\n(Mdn = {p_med:.2f})'],
                       fontsize=24, fontweight='bold')
    d = cohens_d(c_lam, p_lam)
    draw_bracket(ax, 1, 2, 5.0, f'd = {d:.2f}{sig_stars(d)}')

    # (c) Scatter with quadrant dividers
    ax = axes[2]
    np.random.seed(42)
    n_plot = 500
    idx_c = np.random.choice(len(c_lam), min(n_plot, len(c_lam)), replace=False)
    idx_p = np.random.choice(len(p_lam), min(n_plot, len(p_lam)), replace=False)
    ax.scatter(c_lam[idx_c], c_foc[idx_c], color=C_COLOR, alpha=0.4, s=30,
               label='Classical')
    ax.scatter(p_lam[idx_p], p_foc[idx_p], color=P_COLOR, alpha=0.4, s=30,
               label='Pop')

    # Combined median thresholds (absolute, not per-corpus)
    all_lam = np.concatenate([c_lam, p_lam])
    all_foc = np.concatenate([c_foc, p_foc])
    med_lam = np.median(all_lam)
    med_foc = np.median(all_foc)
    ax.axvline(med_lam, color='#555555', linestyle='--', alpha=0.8, linewidth=2.5)
    ax.axhline(med_foc, color='#555555', linestyle='--', alpha=0.8, linewidth=2.5)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    px, py = (xlim[1] - xlim[0]) * 0.02, (ylim[1] - ylim[0]) * 0.02
    fs = 22
    for txt, ha, va, x, y in [
        ('Systematic\ndiatonicism', 'right', 'top', xlim[1] - px, ylim[1] - py),
        ('Textural\ndiatonicism', 'left', 'top', xlim[0] + px, ylim[1] - py),
        ('Chromatic\nexploration', 'right', 'bottom', xlim[1] - px, ylim[0] + py),
        ('Edge of\ntonality', 'left', 'bottom', xlim[0] + px, ylim[0] + py),
    ]:
        ax.text(x, y, txt, ha=ha, va=va, fontsize=fs, fontweight='bold',
                fontstyle='italic', color='#333333', zorder=5)

    ax.set_xlabel('Tonal Connection', fontsize=28, fontweight='bold')
    ax.set_ylabel('Tonal Focus', fontsize=28, fontweight='bold')
    ax.set_title('(c) Tonal Archetypes', fontweight='bold', fontsize=32)
    ax.tick_params(axis='both', labelsize=26)
    ax.legend(fontsize=22, loc='lower center', markerscale=2.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save(fig, "fig2_main_results")


# =========================================================================
# Figure 3 — Dimension characterization (§4.3)
# =========================================================================

def _load_focus_threshold_data():
    """Compute focus at multiple k thresholds for ridge plot."""
    from scipy.stats import entropy as sp_entropy

    df = load_all_tdm_results()
    df = df[df['converged'] == True]
    df = df[(df['lambda'] > 0.05) & (df['lambda'] < 4.95)]

    c_pkg = np.load(RESULTS_DIR / "data/classical_pitch_class_distributions.npz",
                    allow_pickle=True)
    p_pkg = np.load(RESULTS_DIR / "data/lmd_filtered_with_genre.npz",
                    allow_pickle=True)

    c_map = dict(zip(c_pkg['piece_ids'], c_pkg['distributions']))
    p_map = {}
    for pid, d in zip(p_pkg['piece_ids'], p_pkg['distributions']):
        p_map[pid] = d
        p_map[str(pid).replace('.mid', '')] = d

    filt = set()
    for pid in p_pkg['piece_ids']:
        filt.add(str(pid)); filt.add(str(pid).replace('.mid', ''))

    k_values = [2, 3, 4, 5, 6, 7]
    rows = []
    for _, row in df.iterrows():
        pid, corpus, center = row['piece_id'], row['corpus'], row['tonal_center']
        if pd.isna(row['lambda']) or pd.isna(center):
            continue
        if corpus == 'Pop' and str(pid) not in filt:
            continue
        dist = c_map.get(pid) if corpus == 'Classical' else p_map.get(pid)
        if dist is None and corpus == 'Pop':
            dist = p_map.get(str(pid).replace('.mid', ''))
        if dist is None:
            continue
        dn = dist / (dist.sum() + 1e-10)
        ci = int(center)
        if corpus == 'Pop':
            ent = sp_entropy(dn + 1e-10, base=2)
            f3 = sum(dn[max(0, ci - 3):ci + 4])
            if ent > 3.2 or f3 < 0.3:
                continue
        fr = {'piece_id': pid, 'corpus': corpus}
        indices = np.arange(len(dn))
        for k in k_values:
            fr[f'focus_k{k}'] = dn[np.abs(indices - ci) <= k].sum()
        rows.append(fr)
    return pd.DataFrame(rows)


def figure3():
    """Two-panel figure: dumbbell weight profiles + ridge plots."""
    _set_style()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.8)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'axes.titlesize': 20, 'axes.labelsize': 16,
        'xtick.labelsize': 20, 'ytick.labelsize': 17,
    })
    print("Generating Figure 3 (tonal dimensions)...")

    _, _, c_df, _, _, p_df = _load_main_data()
    focus_df = _load_focus_threshold_data()

    fig = plt.figure(figsize=(10, 13))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    # (a) Dumbbell — interval weight profiles
    intervals = ['+P5', '−P5', '+M3', '−M3', '+m3', '−m3']
    col_names = ['+P5', '-P5', '+M3', '-M3', '+m3', '-m3']
    n = len(intervals)
    y_pos = np.arange(n)[::-1]
    offset = 0.13

    c_means = np.array([c_df[c].mean() for c in col_names])
    p_means = np.array([p_df[c].mean() for c in col_names])
    c_sds = np.array([c_df[c].std() for c in col_names])
    p_sds = np.array([p_df[c].std() for c in col_names])

    ax_a.errorbar(c_means, y_pos + offset, xerr=c_sds, fmt='o', color=C_COLOR,
                  markersize=11, capsize=5, linewidth=2, label='Classical',
                  zorder=3, markeredgecolor='white', markeredgewidth=0.8)
    ax_a.errorbar(p_means, y_pos - offset, xerr=p_sds, fmt='s', color=P_COLOR,
                  markersize=11, capsize=5, linewidth=2, label='Pop',
                  zorder=3, markeredgecolor='white', markeredgewidth=0.8)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(intervals, fontsize=21, fontweight='bold')
    ax_a.set_xlabel('Mean Weight (±1 SD)', fontsize=22, fontweight='bold')
    ax_a.set_title('(a) Connection Interval Weights', fontweight='bold', fontsize=24)
    ax_a.legend(loc='lower right', framealpha=0.9, fontsize=13)
    ax_a.grid(axis='x', alpha=0.3); ax_a.grid(axis='y', alpha=0.1)
    for i in range(0, n, 2):
        ax_a.axhspan(y_pos[i] - 0.4, y_pos[i] + 0.4, color='#f0f0f0',
                     alpha=0.5, zorder=0)

    # (b) Ridge — focus distributions at multiple k
    k_values = [2, 3, 5, 7]
    spacing = 0.55
    tick_pos, tick_lab = [], []
    for i, k in enumerate(k_values):
        yo = (len(k_values) - 1 - i) * spacing
        col = f'focus_k{k}'
        c_data = focus_df[focus_df['corpus'] == 'Classical'][col].dropna().values
        p_data = focus_df[focus_df['corpus'] == 'Pop'][col].dropna().values
        x_grid = np.linspace(0, 1.05, 400)
        c_kde = gaussian_kde(c_data, bw_method=0.12)
        p_kde = gaussian_kde(p_data, bw_method=0.12)
        cd, pd_ = c_kde(x_grid), p_kde(x_grid)
        scale = 0.42 / max(cd.max(), pd_.max())
        cd *= scale; pd_ *= scale
        ax_b.fill_between(x_grid, yo, yo + pd_, alpha=0.35, color=P_COLOR, zorder=2+i)
        ax_b.plot(x_grid, yo + pd_, color=P_COLOR, linewidth=1.2, zorder=2+i)
        ax_b.fill_between(x_grid, yo, yo + cd, alpha=0.35, color=C_COLOR, zorder=2+i+0.5)
        ax_b.plot(x_grid, yo + cd, color=C_COLOR, linewidth=1.2, zorder=2+i+0.5)
        ax_b.axhline(yo, color='#ccc', linewidth=0.5, alpha=0.5, zorder=1)
        for data, kde, clr in [(c_data, c_kde, C_COLOR), (p_data, p_kde, P_COLOR)]:
            med = np.median(data)
            h = kde(med)[0] * scale
            ax_b.vlines(med, yo, yo + h, colors=clr, linewidth=1.8, alpha=0.8)
        tick_pos.append(yo + spacing * 0.3)
        tick_lab.append(f'k={k}')

    ax_b.set_xlabel('Tonal Focus', fontsize=22, fontweight='bold')
    ax_b.set_yticks(tick_pos)
    ax_b.set_yticklabels(tick_lab, fontsize=21, fontweight='bold')
    ax_b.set_title('(b) Focus Distributions by $k$', fontweight='bold', fontsize=24)
    ax_b.legend(handles=[Patch(facecolor=C_COLOR, alpha=0.4, label='Classical'),
                         Patch(facecolor=P_COLOR, alpha=0.4, label='Pop')],
                loc='upper right', framealpha=0.9, fontsize=13)

    _save(fig, "fig3_tonal_dimensions")


# =========================================================================
# Figure 4 — Historical trajectory (§4.4)
# =========================================================================

COMPOSER_YEARS = {
    'Sweelinck': 1610, 'Peri': 1600, 'Monteverdi': 1610,
    'Frescobaldi': 1630, 'Schütz': 1640, 'Pergolesi': 1735,
    'Corelli': 1690, 'Couperin': 1710, 'Vivaldi': 1720,
    'Bach': 1730, 'Handel': 1730, 'Scarlatti': 1740, 'Telemann': 1740,
    'ABC': 1700,
    'Cpe Bach': 1760, 'Wf Bach': 1760, 'Jc Bach': 1770,
    'Koželuh': 1790, 'Pleyel': 1790,
    'Haydn': 1780, 'Mozart': 1785, 'Clementi': 1790,
    'Beethoven': 1810, 'Hummel': 1810,
    'Schubert': 1825, 'Mendelssohn': 1840, 'C. Schumann': 1845,
    'Schumann': 1845, 'Chopin': 1840,
    'Liszt': 1860, 'Wagner': 1870, 'Brahms': 1880, 'Bruckner': 1880,
    'Tchaikovsky': 1885, 'Dvořák': 1890, 'Grieg': 1890,
    'Mahler': 1900, 'Rachmaninoff': 1910, 'Medtner': 1920, 'Scriabin': 1910,
    'Debussy': 1905, 'Ravel': 1915, 'Bartók': 1930, 'Stravinsky': 1930,
    'Prokofiev': 1930, 'Shostakovich': 1940, 'Poulenc': 1940, 'Schulhoff': 1930,
}

COMPOSER_ERA = {
    'Sweelinck': 'Baroque', 'Peri': 'Baroque', 'Monteverdi': 'Baroque',
    'Frescobaldi': 'Baroque', 'Schütz': 'Baroque', 'Pergolesi': 'Baroque',
    'Corelli': 'Baroque', 'Couperin': 'Baroque', 'Vivaldi': 'Baroque',
    'Bach': 'Baroque', 'Handel': 'Baroque', 'Scarlatti': 'Baroque',
    'Telemann': 'Baroque', 'ABC': 'Baroque',
    'Cpe Bach': 'Classical', 'Wf Bach': 'Classical', 'Jc Bach': 'Classical',
    'Koželuh': 'Classical', 'Pleyel': 'Classical',
    'Haydn': 'Classical', 'Mozart': 'Classical', 'Clementi': 'Classical',
    'Beethoven': 'Classical', 'Hummel': 'Classical',
    'Schubert': 'Romantic', 'Mendelssohn': 'Romantic', 'C. Schumann': 'Romantic',
    'Schumann': 'Romantic', 'Chopin': 'Romantic', 'Liszt': 'Romantic',
    'Wagner': 'Romantic', 'Brahms': 'Romantic', 'Bruckner': 'Romantic',
    'Tchaikovsky': 'Romantic', 'Dvořák': 'Romantic', 'Grieg': 'Romantic',
    'Mahler': 'Romantic', 'Rachmaninoff': 'Romantic', 'Medtner': 'Romantic',
    'Scriabin': 'Romantic',
    'Debussy': 'Early 20th C.', 'Ravel': 'Early 20th C.', 'Bartók': 'Early 20th C.',
    'Stravinsky': 'Early 20th C.', 'Prokofiev': 'Early 20th C.', 'Shostakovich': 'Early 20th C.',
    'Poulenc': 'Early 20th C.', 'Schulhoff': 'Early 20th C.',
}

ERA_PALETTE = {
    'Baroque': '#0072B2', 'Classical': '#E69F00',
    'Romantic': '#009E73', 'Early 20th C.': '#D55E00',
}


# DLC corpus directory → canonical composer name
CORPUS_TO_COMPOSER = {
    'ABC': 'ABC',
    'bach_en_fr_suites': 'Bach', 'bach_solo': 'Bach',
    'bartok_bagatelles': 'Bartók',
    'beethoven_piano_sonatas': 'Beethoven',
    'c_schumann_lieder': 'C. Schumann',
    'chopin_mazurkas': 'Chopin',
    'corelli': 'Corelli',
    'couperin_clavecin': 'Couperin', 'couperin_concerts': 'Couperin',
    'cpe_bach_keyboard': 'Cpe Bach',
    'debussy_suite_bergamasque': 'Debussy',
    'dvorak_silhouettes': 'Dvořák',
    'frescobaldi_fiori_musicali': 'Frescobaldi',
    'grieg_lyric_pieces': 'Grieg',
    'handel_keyboard': 'Handel',
    'jc_bach_sonatas': 'Jc Bach',
    'kleine_geistliche_konzerte': 'Schütz',
    'kozeluh_sonatas': 'Koželuh',
    'liszt_pelerinage': 'Liszt',
    'mahler_kindertotenlieder': 'Mahler',
    'medtner_tales': 'Medtner',
    'mendelssohn_quartets': 'Mendelssohn',
    'monteverdi_madrigals': 'Monteverdi',
    'mozart_piano_sonatas': 'Mozart',
    'pergolesi_stabat_mater': 'Pergolesi',
    'peri_euridice': 'Peri',
    'pleyel_quartets': 'Pleyel',
    'poulenc_mouvements_perpetuels': 'Poulenc',
    'rachmaninoff_piano': 'Rachmaninoff',
    'ravel_piano': 'Ravel',
    'scarlatti_sonatas': 'Scarlatti',
    'schubert_winterreise': 'Schubert',
    'schulhoff_suite_dansante_en_jazz': 'Schulhoff',
    'schumann_kinderszenen': 'Schumann', 'schumann_liederkreis': 'Schumann',
    'sweelinck_keyboard': 'Sweelinck',
    'tchaikovsky_seasons': 'Tchaikovsky',
    'wagner_overtures': 'Wagner',
    'wf_bach_sonatas': 'Wf Bach',
}


def _extract_composer_from_path(file_path):
    """Extract composer from DLC path: .../distant_listening_corpus/{corpus}/notes/..."""
    parts = Path(str(file_path)).parts
    for i, part in enumerate(parts):
        if part == 'distant_listening_corpus' and i + 1 < len(parts):
            return CORPUS_TO_COMPOSER.get(parts[i + 1], 'Unknown')
    return 'Unknown'


def _load_classical_with_composers():
    """Load classical TDM + distributions, assign composers/eras/years.

    Caches the expensive metadata extraction to results/cache/.
    """
    import json as _json

    cache_dir = RESULTS_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "classical_composer_meta.csv"

    tdm = load_all_tdm_results()
    df = tdm[(tdm['corpus'] == 'Classical') & (tdm['converged'] == True)].copy()

    if cache_path.exists():
        print("  Loading cached composer metadata...")
        df_meta = pd.read_csv(cache_path)
    else:
        print("  Building composer metadata (first run, will be cached)...")
        c_pkg = np.load(RESULTS_DIR / "data/classical_pitch_class_distributions.npz",
                        allow_pickle=True)
        focus_vals = []
        for dist in c_pkg['distributions']:
            focus_vals.append(max(np.sum(dist[i:i+7]) for i in range(len(dist) - 6)))
        composers = [_extract_composer_from_path(fp) for fp in c_pkg['file_paths']]
        df_meta = pd.DataFrame({
            'filename': c_pkg['piece_ids'],
            'focus': focus_vals,
            'composer': composers,
        })
        df_meta.to_csv(cache_path, index=False)
        print(f"  Cached to {cache_path}")

    df['join_key'] = df['piece_id'].astype(str).str.strip()
    df_meta['join_key'] = df_meta['filename'].astype(str).str.strip()
    df = pd.merge(df, df_meta, on='join_key', suffixes=('_tdm', '_meta'))

    df['year'] = df['composer'].map(COMPOSER_YEARS)
    df['era'] = df['composer'].map(COMPOSER_ERA).fillna('Other')
    return df


def figure4():
    """Historical trajectory: composer centroids with era KDE contours."""
    _set_style()
    print("Generating Figure 4 (historical trajectory)...")
    import matplotlib.transforms as transforms
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    df = _load_classical_with_composers()
    df = df.dropna(subset=['year'])

    sns.set_style("white")
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})
    fig, ax = plt.subplots(figsize=(18, 14))

    eras_ordered = ['Baroque', 'Classical', 'Romantic', 'Early 20th C.']
    sns.scatterplot(data=df, x='lambda', y='focus', hue='era',
                    palette=ERA_PALETTE, alpha=0.05, s=20, legend=False, ax=ax)

    centroids = []
    for era in eras_ordered:
        sub = df[df['era'] == era]
        if len(sub) < 5:
            continue
        centroids.append((sub['lambda'].mean(), sub['focus'].mean()))
        try:
            sns.kdeplot(data=sub, x='lambda', y='focus', fill=True, alpha=0.1,
                        color=ERA_PALETTE[era], thresh=0.2, levels=2, ax=ax,
                        zorder=2, cut=3)
            sns.kdeplot(data=sub, x='lambda', y='focus', fill=False,
                        linewidths=2, linestyles='--', color=ERA_PALETTE[era],
                        thresh=0.2, levels=2, ax=ax, zorder=3, cut=3)
        except Exception:
            pass

    for i in range(len(centroids) - 1):
        ax.annotate("", xy=centroids[i+1], xytext=centroids[i],
                    arrowprops=dict(arrowstyle="-|>,head_length=1.2,head_width=0.6",
                                    color="#555555", lw=5, connectionstyle="arc3,rad=0.1"),
                    zorder=15)

    comp_stats = df.groupby('composer')[['lambda', 'focus']].mean()
    comp_stats['era'] = comp_stats.index.map(COMPOSER_ERA).fillna('Unknown')

    ax.set_xlim(0, 5.0); ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Tonal Connection", fontsize=32, fontweight='bold')
    ax.set_ylabel("Tonal Focus", fontsize=32, fontweight='bold')
    ax.tick_params(axis='both', labelsize=28)
    fig.subplots_adjust(left=0.22)

    all_comps = []
    for comp, row in comp_stats.iterrows():
        color = ERA_PALETTE.get(row['era'], '#999')
        ax.scatter(row['lambda'], row['focus'], color=color, s=150, marker='o',
                   zorder=10, edgecolor='white', linewidth=2, alpha=0.9)
        all_comps.append((comp, row['lambda'], row['focus'], row['era']))

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_range, y_range = xlim[1] - xlim[0], ylim[1] - ylim[0]

    left_comps = [(c, x, y, e) for c, x, y, e in all_comps if e in ('Baroque', 'Romantic')]
    right_comps = [(c, x, y, e) for c, x, y, e in all_comps if e in ('Classical', 'Early 20th C.')]
    era_order = {e: i for i, e in enumerate(eras_ordered)}
    key_fn = lambda c: (era_order.get(c[3], 99), -c[2])
    left_comps.sort(key=key_fn); right_comps.sort(key=key_fn)

    def evenly_spaced(n, lo, hi):
        if n <= 1:
            return [(lo + hi) / 2] * max(n, 1)
        return [lo + i * (hi - lo) / (n - 1) for i in range(n)]

    left_x = xlim[0] + 0.3
    left_ys = evenly_spaced(len(left_comps), ylim[1] - 0.01*y_range, ylim[0] + 0.04*y_range)
    right_x = xlim[1] - 0.15
    step = (left_ys[0] - left_ys[-1]) / max(len(left_comps) - 1, 1) if left_comps else 0.025
    right_ys = [ylim[1] - 0.03*y_range - i*step for i in range(len(right_comps))]

    def place(comp, cx, cy, era, tx, ty, ha, rad):
        color = ERA_PALETTE.get(era, '#999')
        rgb = mcolors.to_rgb(color)
        bg = tuple(c*0.35 + 0.65 for c in rgb)
        ax.annotate(comp, (cx, cy), xytext=(tx, ty), textcoords='data',
                    fontsize=22, fontweight='bold', ha=ha, va='center',
                    bbox=dict(facecolor=bg, alpha=0.85, edgecolor=color,
                              linewidth=1.2, pad=2, boxstyle='round,pad=0.3'),
                    arrowprops=dict(arrowstyle='-', color='#bbbbbb', lw=0.7,
                                    connectionstyle=f'arc3,rad={rad}'),
                    zorder=15)

    for (c, cx, cy, e), ly in zip(left_comps, left_ys):
        place(c, cx, cy, e, left_x, ly, 'left', 0.2)
    for (c, cx, cy, e), ly in zip(right_comps, right_ys):
        place(c, cx, cy, e, right_x, ly, 'right', -0.2)

    leg = ax.legend(
        handles=[Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=ERA_PALETTE[e], markersize=12, label=e)
                 for e in eras_ordered],
        loc='lower right', frameon=True, title="Stylistic Era",
        prop={'weight': 'bold', 'size': 22})
    leg.get_title().set_fontsize(24)
    leg.get_title().set_fontweight('bold')
    sns.despine(trim=True)

    _save(fig, "fig4_historical_trajectory")


# =========================================================================
# Figure 5 — Genre breakdown (App A)
# =========================================================================

def figure5():
    """Genre scatter: mean λ vs focus for 12 pop genres."""
    _set_style()
    print("Generating Figure 5 (genre breakdown)...")

    tdm = load_all_tdm_results()
    df = tdm[tdm['corpus'] == 'Pop'].copy()

    dist_pkg = np.load(RESULTS_DIR / "data/lmd_filtered_with_genre.npz",
                       allow_pickle=True)
    pid_to_idx = {}
    for i, pid in enumerate(dist_pkg['piece_ids']):
        pid_to_idx[str(pid).replace('.mid', '')] = i

    focus_vals = []
    for _, row in df.iterrows():
        pid = str(row['piece_id']).replace('.mid', '')
        tc = row['tonal_center']
        if pid in pid_to_idx and not pd.isna(tc):
            d = dist_pkg['distributions'][pid_to_idx[pid]]
            focus_vals.append(compute_tonal_focus(d, tc, k=3))
        else:
            focus_vals.append(np.nan)
    df['focus'] = focus_vals

    stats_rows = []
    for genre in df['genre'].unique():
        sub = df[df['genre'] == genre]
        lv, fv = sub['lambda'].dropna(), sub['focus'].dropna()
        stats_rows.append({
            'Genre': genre, 'N': len(sub),
            'lambda_mean': lv.mean(), 'focus_mean': fv.mean(),
            'lambda_ci': 1.96 * lv.std() / np.sqrt(len(lv)),
            'focus_ci': 1.96 * fv.std() / np.sqrt(len(fv)),
        })
    sdf = pd.DataFrame(stats_rows).sort_values('N', ascending=False)

    plt.rcParams.update({'font.size': 18, 'mathtext.fontset': 'stix'})
    fig, ax = plt.subplots(figsize=(12, 10))

    genre_colors = {
        'Rock': '#1f77b4', 'Pop': '#ff7f0e', 'Country': '#2ca02c',
        'Electronic': '#d62728', 'RnB': '#9467bd', 'Metal': '#8c564b',
        'Rap': '#e377c2', 'Latin': '#7f7f7f', 'Reggae': '#bcbd22',
        'Folk': '#17becf', 'World': '#aec7e8', 'Punk': '#ffbb78',
    }
    offsets = {
        'Folk': (-70, 10), 'Punk': (-70, 0), 'Reggae': (15, 10),
        'World': (15, 0), 'Metal': (15, 0), 'Rock': (-70, 0),
        'Electronic': (-115, 0), 'Country': (15, -18), 'Pop': (15, 0),
        'Rap': (-60, 0), 'Latin': (15, 10), 'RnB': (-60, -15),
    }

    for _, row in sdf.iterrows():
        g = row['Genre']
        ax.errorbar(row['lambda_mean'], row['focus_mean'],
                    xerr=row['lambda_ci'], yerr=row['focus_ci'],
                    fmt='o', markersize=16 + np.log(row['N']) * 4,
                    color=genre_colors.get(g, '#333'), capsize=5, capthick=1.5,
                    alpha=0.5, elinewidth=1.5, markeredgecolor='white',
                    markeredgewidth=2, zorder=5)
        off = offsets.get(g, (15, 0))
        ax.annotate(g, (row['lambda_mean'], row['focus_mean']),
                    xytext=off, textcoords='offset points',
                    fontsize=24, fontweight='bold', va='center',
                    color=genre_colors.get(g, '#333'), zorder=20)

    ax.set_xlabel('Tonal Connection', fontsize=28, fontweight='bold')
    ax.set_ylabel('Tonal Focus', fontsize=28, fontweight='bold')
    ax.set_title('Tonal Organisation of Pop Music Genres', fontsize=32,
                 fontweight='bold', pad=20)
    ax.set_xlim(1.5, 2.5); ax.set_ylim(0.62, 0.82)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=26)
    plt.tight_layout()
    _save(fig, "fig5_genre_breakdown")


# =========================================================================
# Figure 6 — Composer boxplots (App A)
# =========================================================================

def figure6():
    """Composer-level boxplots colored by era."""
    _set_style()
    print("Generating Figure 6 (composer boxplots)...")

    df = _load_classical_with_composers()
    df = df[df['era'] != 'Other'].copy()

    eras = ['Baroque', 'Classical', 'Romantic', 'Early 20th C.']
    df['era'] = pd.Categorical(df['era'], categories=eras, ordered=True)
    df = df.sort_values(['era', 'composer'])

    plt.rcParams.update({'font.size': 18, 'mathtext.fontset': 'stix'})
    era_pal = {'Baroque': '#1f77b4', 'Classical': '#ff7f0e',
               'Romantic': '#d62728', 'Early 20th C.': '#9467bd'}

    n_comp = df['composer'].nunique()
    fig_h = max(12, n_comp * 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_h))

    sns.boxplot(data=df, y='composer', x='lambda', hue='era', dodge=False,
                ax=axes[0], palette=era_pal, legend=True)
    axes[0].set_title("Tonal Connection", fontsize=32, fontweight='bold')
    axes[0].set_xlabel(None); axes[0].set_ylabel(None)
    axes[0].tick_params(axis='y', labelsize=26)
    axes[0].tick_params(axis='x', labelsize=26)
    if axes[0].get_legend():
        axes[0].get_legend().set_title('Era')
        for t in axes[0].get_legend().get_texts():
            t.set_fontsize(20)

    sns.boxplot(data=df, y='composer', x='focus', hue='era', dodge=False,
                ax=axes[1], palette=era_pal, legend=False)
    axes[1].set_title("Tonal Focus", fontsize=32, fontweight='bold')
    axes[1].set_xlabel(None); axes[1].set_ylabel(None)
    axes[1].tick_params(axis='y', labelsize=26)
    axes[1].tick_params(axis='x', labelsize=26)

    plt.suptitle("Tonal Organization Across Classical Composers",
                 fontsize=36, fontweight='bold')
    plt.tight_layout()
    _save(fig, "fig6_composer_boxplots")


# =========================================================================
# Main
# =========================================================================

FIGURE_MAP = {
    2: ('§4.1 Main results', figure2),
    3: ('§4.3 Dimension characterization', figure3),
    4: ('§4.4 Historical trajectory', figure4),
    5: ('App A: Genre breakdown', figure5),
    6: ('App A: Composer boxplots', figure6),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all paper figures")
    parser.add_argument('--fig', nargs='*', type=int,
                        help="Figure numbers (2-6). Default: all")
    args = parser.parse_args()

    selected = args.fig if args.fig else sorted(FIGURE_MAP.keys())
    for n in selected:
        if n in FIGURE_MAP:
            name, fn = FIGURE_MAP[n]
            print(f"\n{'#' * 60}")
            print(f"# Figure {n}: {name}")
            print(f"{'#' * 60}")
            fn()
    print("\nDone.")
