# Diachronic Modeling of Tonal Coherence on the Tonnetz Across Classical and Popular Repertoires

Code and data for **"Diachronic Modeling of Tonal Coherence on the Tonnetz Across Classical and Popular Repertoires"** (Xu, Hall & Rohrmeier), under review at SMC 2026.

## Overview

This project compares tonal organization strategies between Western classical music (1680–1920) and popular music (1950–2020) using two complementary metrics derived from the Tonal Diffusion Model (TDM):

- **Tonal Connection (λ):** How far pitch content traverses structured intervallic pathways on the Tonnetz
- **Tonal Focus:** How concentrated pitch content is around the tonal center on the line-of-fifths

## Repository Structure

```
src/
├── tonal_focus.py              # ★ Standalone: Tonal Focus measure (§3.4.1)
├── tonal_connection.py         # ★ Standalone: Tonal Connection / TDM fitting (§3.4.2)
├── models/
│   └── tonal_diffusion.py      # Tonal Diffusion Model (§3.4)
├── data/
│   ├── extract.py              # Unified extraction: classical/pop35d/pop12d/windowed (§3.1–3.2)
│   └── filter_lmd.py           # LMD quality filtering pipeline (§3.1)
├── experiments/
│   └── tdm_fitting.py          # TDM parameter fitting (§3.4)
├── utils/
│   ├── loaders.py              # Data loading, path constants
│   ├── metrics.py              # TDM metric calculations
│   ├── stats.py                # Statistical helpers (effect sizes, annotation formatting)
│   └── download_lmd.py         # Data acquisition helper
├── generate_figures.py         # All paper figures (Figures 2–6, ordered by §4 + App A)
└── generate_tables.py          # All appendix tables (Tables B.1–B.5, ordered by App B)
figures/                        # Generated publication figures
results/                        # Pre-computed distributions and TDM fitting results
```

## Setup

```bash
pip install -r requirements.txt
```

### Data

The analysis uses two corpora (not included due to licensing):

- **[Distant Listening Corpus (DLC)](https://github.com/DCMLab/distant_listening_corpus)** — 1,326 classical keyboard works (1680–1920) with expert annotations. After TDM convergence filtering, ~1,280 pieces are used in the final analysis.
- **[Lakh MIDI Dataset (LMD)](https://colinraffel.com/projects/lmd/)** — Popular music MIDI files, filtered to 1,569 pieces across 12 genres via quality thresholds (pitch entropy, single-pitch dominance, tonal focus; see §3.1 and Appendix A).

Place datasets in `data/` following the structure expected by `src/utils/loaders.py`.

## Running the Analysis

```bash
# 1. Extract pitch-class distributions from raw corpora
python -m src.data.extract all

# 2. Filter LMD corpus
python -m src.data.filter_lmd

# 3. Fit TDM parameters
python -m src.experiments.tdm_fitting

# 4. Generate all figures (or select: --fig 2 4)
python -m src.generate_figures

# 5. Generate all appendix tables (or select: --table 1 3)
python -m src.generate_tables
```

## Computing Tonal Focus and Tonal Connection

The two core measures can be computed directly from pitch-class distributions (the `.npz` files produced by step 1 above).

### Tonal Focus (§3.4.1)

Measures how concentrated pitch content is around the tonal center on the line-of-fifths.

```bash
# From extracted distributions
python -m src.tonal_focus results/data/classical_pitch_class_distributions.npz

# With custom k (neighborhood radius, default=3)
python -m src.tonal_focus results/data/lmd_filtered_with_genre.npz --k 5
```

**As a library:**
```python
from src.tonal_focus import compute_tonal_focus, compute_tonal_focus_batch
import numpy as np

# Single piece: distribution (35-D or 12-D) + tonal center index
focus = compute_tonal_focus(distribution, tonal_center=0, k=3)

# Batch: array of distributions + array of centers
results = compute_tonal_focus_batch(distributions, centers, k=3)
```

### Tonal Connection (§3.4.2)

Measures how far pitch content traverses structured intervallic pathways, via the TDM λ parameter.

```bash
# From extracted distributions (10 pieces, 2 random starts for speed)
python -m src.tonal_connection results/data/classical_pitch_class_distributions.npz \
    --max-pieces 10 --n-starts 2

# Full corpus (slower — fits MLE for each piece)
python -m src.tonal_connection results/data/classical_pitch_class_distributions.npz

# Save results to CSV
python -m src.tonal_connection results/data/classical_pitch_class_distributions.npz \
    --output results/tonal_connection_output.csv
```

**As a library:**
```python
from src.tonal_connection import compute_tonal_connection
import numpy as np

# Single piece: distribution (35-D) + tonal center index
result = compute_tonal_connection(distribution, tonal_center=0, n_starts=5)
# Returns: {'lambda': float, 'weights': array, 'fifth_dominance': float, ...}
```

### Full Pipeline (Raw Data → Measures)

```bash
# Step 1: Extract pitch-class distributions from raw corpora
python -m src.data.extract classical   # or: pop35d, pop12d, all

# Step 2: Compute tonal focus
python -m src.tonal_focus results/data/classical_pitch_class_distributions.npz

# Step 3: Compute tonal connection
python -m src.tonal_connection results/data/classical_pitch_class_distributions.npz
```

## Pre-computed Results

The repository includes pre-computed intermediate data so that figures and tables can be regenerated without access to the raw corpora:

- `results/data/` — Extracted pitch-class distributions (`.npz`)
- `results/tdm_analysis/` — Fitted TDM parameters (`.csv`)

These are sufficient to run steps 4–5 of the pipeline (figure and table generation).

## Supplementary Material

The paper appendix (corpus details, robustness checks, non-parametric effect sizes, key estimation validation, and detailed genre/composer variations) is provided as a separate document to respect submission page limits.

- **PDF:** [`supplementary_material.pdf`](supplementary_material.pdf)
- **LaTeX source:** `paper/SMC Paper/supplementary_material.tex`

## Citation

This work is currently under review. If you use this code or method, please cite:

```bibtex
@unpublished{xu2026diachronic,
  title  = {Diachronic Modeling of Tonal Coherence on the Tonnetz Across Classical and Popular Repertoires},
  author = {Xu, Weilun and Hall, Edward and Rohrmeier, Martin},
  note   = {Under review at the Sound and Music Computing Conference (SMC)},
  year   = {2026}
}
```

## License

See [LICENSE](LICENSE) for details.
