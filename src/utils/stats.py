"""
Shared statistical utility functions for tonal coherence analyses.
"""

import numpy as np
from scipy import stats as sp_stats
from scipy.stats import kurtosis as sp_kurtosis


# ---------------------------------------------------------------------------
# Effect Size Functions
# ---------------------------------------------------------------------------

def cohens_d(group1, group2):
    """Compute Cohen's d (pooled standard deviation)."""
    g1, g2 = np.asarray(group1), np.asarray(group2)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def cliffs_delta(group1, group2):
    """Compute Cliff's delta (non-parametric effect size)."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = sp_stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return (2 * u) / (n1 * n2) - 1


def bootstrap_ci_d(a, b, n_boot=2000, ci=0.95):
    """Bootstrap confidence interval for Cohen's d."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    ds = []
    for _ in range(n_boot):
        sa = a[np.random.randint(0, len(a), len(a))]
        sb = b[np.random.randint(0, len(b), len(b))]
        ds.append(cohens_d(sa, sb))
    lo = (1 - ci) / 2
    return np.nanpercentile(ds, lo * 100), np.nanpercentile(ds, (1 - lo) * 100)


def rank_biserial(group1, group2):
    """Compute rank-biserial correlation from Mann-Whitney U."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = sp_stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return 1 - (2 * u) / (n1 * n2)


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------


def compute_kurtosis_from_weights(row):
    """Compute excess kurtosis of TDM interval weight vector (Fisher)."""
    intervals = ['+P5', '-P5', '+M3', '-M3', '+m3', '-m3']
    w = np.array([float(row[c]) for c in intervals if c in row.index])
    if w.sum() == 0:
        return np.nan
    w = w / w.sum()
    return sp_kurtosis(w, fisher=True)


def compute_kurtosis_lof(row):
    """Compute excess kurtosis for 12-D TDM weights using line-of-fifths mapping."""
    map_lof = {
        '+P5': 1, '-P5': -1,
        '+M3': 4, '-M3': -4,
        '+m3': -3, '-m3': 3,
    }
    xs, ws = [0], []
    current_sum = 0
    for col, dist in map_lof.items():
        if col in row:
            val = float(row[col])
            xs.append(dist)
            ws.append(val)
            current_sum += val
    ws.insert(0, max(0, 1.0 - current_sum))
    xs, ws = np.array(xs), np.array(ws)
    if ws.sum() == 0:
        return np.nan
    ws = ws / ws.sum()
    mean = np.sum(xs * ws)
    var = np.sum(ws * (xs - mean) ** 2)
    std = np.sqrt(var)
    if std < 0.1:
        return np.nan
    m4 = np.sum(ws * (xs - mean) ** 4)
    return m4 / (std ** 4) - 3


def compute_entropy_from_weights(row):
    """Compute Shannon entropy (bits) of interval weight vector."""
    cols = [c for c in row.index
            if c.startswith('+') or c.startswith('-') or c in ('P1', 'tt')]
    probs = row[cols].astype(float).values
    total = np.sum(probs)
    if total == 0:
        return np.nan
    probs = probs / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


# ---------------------------------------------------------------------------
# Plotting Annotation Helpers
# ---------------------------------------------------------------------------

def sig_stars(d):
    """Return significance stars based on effect size magnitude."""
    d = abs(d)
    if d >= 0.8:
        return '***'
    elif d >= 0.5:
        return '**'
    elif d >= 0.2:
        return '*'
    return ''


def draw_bracket(ax, x1, x2, y, text, lw=2.5, color='#333333'):
    """Draw a bracket between x1 and x2 at height y with text above."""
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h],
            lw=lw, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 0.5, text, ha='center', va='bottom',
            fontsize=14, fontweight='bold', color=color)
