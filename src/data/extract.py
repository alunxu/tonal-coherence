#!/usr/bin/env python3
"""
Unified data extraction pipeline (§3.1–3.2).

Subcommands:
  classical   - 35-D distributions from DLC (TSV files)
  pop35d      - 35-D distributions from LMD (Partitura pitch spelling)
  pop12d      - 12-D distributions from LMD (MIDI pitch class)
  windowed    - Windowed distributions (first 16 bars) for both corpora
  all         - Run all of the above
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from fractions import Fraction
from collections import Counter
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.loaders import DATA_DIR, RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOF_SIZE = 35
CENTER_IDX = 17  # C natural = index 17
LMD_PATH = DATA_DIR / "lmd_aligned"
DLC_PATH = DATA_DIR / "distant_listening_corpus"


# =========================================================================
# Classical 35-D (§3.2)
# =========================================================================

def _process_tsv(tsv_path):
    """Extract 35-D distribution from a single DLC notes.tsv file."""
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if 'tpc' not in df.columns or 'duration' not in df.columns:
            return None
        dist = np.zeros(LOF_SIZE)
        valid = df[df['tpc'].notna()]
        for _, row in valid.iterrows():
            tpc = int(row['tpc'])
            try:
                dur = float(Fraction(str(row['duration'])))
            except Exception:
                continue
            idx = tpc + CENTER_IDX
            if 0 <= idx < LOF_SIZE:
                dist[idx] += dur
        if dist.sum() > 0:
            dist /= dist.sum()
        try:
            if 'onset' in df.columns:
                onsets = df['onset'].apply(lambda x: float(Fraction(str(x))))
                durs = df['duration'].apply(lambda x: float(Fraction(str(x))))
                piece_len = float((onsets + durs).max())
            else:
                piece_len = float(
                    valid['duration'].apply(
                        lambda x: float(Fraction(str(x)))).sum())
        except Exception:
            piece_len = 0.0
        return {
            'file_path': str(tsv_path),
            'distribution': dist,
            'piece_id': tsv_path.stem.replace('.notes', ''),
            'total_duration': piece_len,
        }
    except Exception:
        return None


def extract_classical_35d():
    """Extract 35-D pitch class distributions from the Distant Listening Corpus."""
    print("=" * 60)
    print("EXTRACTING CLASSICAL 35-D (DLC)")
    print("=" * 60)
    files = list(DLC_PATH.rglob("*.notes.tsv"))
    if not files:
        print("No TSV files found!")
        return
    print(f"Found {len(files)} files.")
    with mp.Pool() as pool:
        results = list(tqdm(pool.imap(_process_tsv, files), total=len(files)))
    valid = [r for r in results if r is not None]
    print(f"Processed {len(valid)} files.")
    out = OUTPUT_DIR / "classical_pitch_class_distributions.npz"
    np.savez(out,
             distributions=np.array([r['distribution'] for r in valid]),
             piece_ids=np.array([r['piece_id'] for r in valid]),
             file_paths=np.array([r['file_path'] for r in valid]),
             durations=np.array([r['total_duration'] for r in valid]))
    print(f"Saved to {out}")


# =========================================================================
# Pop 35-D via Partitura (§3.2)
# =========================================================================

_STEP_TO_LOF = {
    'C': 17, 'D': 19, 'E': 21, 'F': 16, 'G': 18, 'A': 20, 'B': 22,
}


def _get_lof_position(step, alter):
    if step not in _STEP_TO_LOF:
        return None
    return _STEP_TO_LOF[step] + alter * 7


def _process_midi_35d(midi_file):
    """Extract 35-D from a single MIDI via Partitura spelling."""
    try:
        import partitura
        import pretty_midi
        score = partitura.load_score(str(midi_file))
        all_notes = []
        for part in score.parts:
            try:
                partitura.musicanalysis.estimate_spelling(part)
            except Exception:
                continue
            all_notes.extend(part.notes)
        if not all_notes:
            return None
        dist = np.zeros(35)
        for note in all_notes:
            if not hasattr(note, 'step') or not hasattr(note, 'alter'):
                continue
            alter = note.alter if note.alter is not None else 0
            pos = _get_lof_position(note.step, alter)
            if pos is not None and 0 <= pos < 35:
                dist[pos] += note.duration
        if dist.sum() > 0:
            dist /= dist.sum()
            try:
                pm = pretty_midi.PrettyMIDI(str(midi_file))
                dur_beats = len(pm.get_beats())
            except Exception:
                dur_beats = 0
            return (str(midi_file.name), dist, dur_beats)
        return None
    except Exception:
        return None


def extract_pop_35d():
    """Extract 35-D distributions from LMD using Partitura pitch spelling."""
    print("=" * 60)
    print("EXTRACTING POP 35-D (PARTITURA)")
    print("=" * 60)
    midi_files = (list(LMD_PATH.glob("**/*.mid"))
                  + list(LMD_PATH.glob("**/*.midi")))
    print(f"Found {len(midi_files)} MIDI files")
    max_workers = max(1, int(mp.cpu_count() * 0.8))
    print(f"Using {max_workers} workers...")
    all_dists, all_ids, all_durs = [], [], []
    chunk_size = 5000
    out = RESULTS_DIR / "lmd_pitch_class_distributions_35d_partitura.npz"
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for res in tqdm(ex.map(_process_midi_35d, midi_files),
                        total=len(midi_files)):
            if res is not None:
                pid, dist, dur = res
                all_ids.append(pid)
                all_dists.append(dist)
                all_durs.append(dur)
            if len(all_dists) % chunk_size == 0 and all_dists:
                np.savez_compressed(
                    out,
                    piece_ids=np.array(all_ids),
                    distributions=np.array(all_dists),
                    durations=np.array(all_durs))
    if all_dists:
        np.savez_compressed(
            out,
            piece_ids=np.array(all_ids),
            distributions=np.array(all_dists),
            durations=np.array(all_durs))
    print(f"Processed {len(all_dists)} files → {out}")


# =========================================================================
# Pop 12-D (Appendix B robustness)
# =========================================================================

def _process_midi_12d(midi_path):
    """Extract 12-D pitch class distribution from a single MIDI file."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        if pm.get_end_time() < 10 or pm.get_end_time() > 600:
            return None
        pc = np.zeros(12)
        total = 0
        for inst in pm.instruments:
            if not inst.is_drum:
                for n in inst.notes:
                    w = (n.end - n.start) * (n.velocity / 127.0)
                    pc[n.pitch % 12] += w
                    total += w
        if total > 0:
            pc /= total
        return {
            'file_path': str(midi_path),
            'distribution': pc,
            'duration': len(pm.get_beats()),
        }
    except Exception:
        return None


def extract_pop_12d():
    """Extract 12-D pitch class distributions from LMD MIDI files."""
    print("=" * 60)
    print("EXTRACTING POP 12-D")
    print("=" * 60)
    files = list(LMD_PATH.rglob("*.mid"))
    if not files:
        print("No MIDI files found!")
        return
    print(f"Found {len(files)} MIDI files.")
    max_workers = max(1, int(mp.cpu_count() * 0.8))
    with mp.Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap(_process_midi_12d, files),
                            total=len(files)))
    valid = [r for r in results if r is not None]
    print(f"Processed {len(valid)} files.")
    out = OUTPUT_DIR / "lmd_pitch_class_distributions.npz"
    np.savez(out,
             distributions=np.array([r['distribution'] for r in valid]),
             file_paths=[r['file_path'] for r in valid],
             durations=np.array([r['duration'] for r in valid]))
    print(f"Saved to {out}")


# =========================================================================
# Windowed Extraction — first 16 bars (Appendix B robustness)
# =========================================================================

def extract_windowed_classical():
    """Extract 35-D distributions for the first 16 measures of DLC pieces."""
    print("=" * 60)
    print("EXTRACTING CLASSICAL WINDOWED (16 BARS)")
    print("=" * 60)
    files = list(DLC_PATH.rglob("*.notes.tsv"))
    print(f"Found {len(files)} files")
    results = []
    for fpath in tqdm(files):
        try:
            df = pd.read_csv(fpath, sep='\t')
            if 'mn' not in df.columns or 'tpc' not in df.columns:
                continue
            df['mn_numeric'] = pd.to_numeric(df['mn'], errors='coerce')
            window = df[df['mn_numeric'] <= 16]
            if window.empty:
                continue
            tpcs = window['tpc'].dropna().astype(int)
            counts = Counter(tpcs)
            dist = np.zeros(LOF_SIZE)
            for tpc_val, count in counts.items():
                idx = tpc_val + CENTER_IDX
                if 0 <= idx < LOF_SIZE:
                    dist[idx] = count
            results.append({
                'piece_id': fpath.name.replace(".notes.tsv", ""),
                'distribution': dist,
            })
        except Exception:
            pass
    print(f"Extracted {len(results)} pieces.")
    out = OUTPUT_DIR / "classical_windowed_16bar.npz"
    np.savez(out,
             piece_ids=[r['piece_id'] for r in results],
             distributions=np.array([r['distribution'] for r in results]))
    print(f"Saved to {out}")


def extract_windowed_pop():
    """Extract 35-D distributions for the first ~16 bars of filtered LMD."""
    import partitura
    print("=" * 60)
    print("EXTRACTING POP WINDOWED (16 BARS / 32 SEC)")
    print("=" * 60)
    try:
        pkg = np.load(RESULTS_DIR / "data/lmd_filtered_with_genre.npz",
                       allow_pickle=True)
        target_ids = set(pid.replace('.mid', '') for pid in pkg['piece_ids'])
        print(f"Targeting {len(target_ids)} filtered pieces.")
        lmd_pkg = np.load(
            RESULTS_DIR / "data/lmd_pitch_class_distributions.npz",
            allow_pickle=True)
        id_to_path = {Path(str(p)).stem: str(p)
                      for p in lmd_pkg['file_paths']}
        process_list = [(pid, id_to_path[pid])
                        for pid in target_ids if pid in id_to_path]
        print(f"Found paths for {len(process_list)} pieces.")
    except FileNotFoundError as e:
        print(f"Error loading metadata: {e}")
        return

    step_to_tpc = {'C': 0, 'D': 2, 'E': 4, 'F': -1, 'G': 1, 'A': 3, 'B': 5}
    results = []
    cutoff_beats = 64.0
    for pid, rel_path in tqdm(process_list):
        try:
            full_path = Path(rel_path)
            if not full_path.exists():
                full_path = DATA_DIR / rel_path
            if not full_path.exists():
                continue
            score = partitura.load_score(str(full_path))
            part = score[0]
            na = part.note_array(include_time_signature=True,
                                  include_pitch_spelling=True)
            if 'onset_beat' not in na.dtype.names:
                continue
            window = na[na['onset_beat'] <= cutoff_beats]
            if len(window) < 10:
                continue
            if ('step' not in window.dtype.names
                    or 'alter' not in window.dtype.names):
                continue
            dist = np.zeros(35)
            for note in window:
                step, alter = note['step'], note['alter']
                dur = (note['duration_beat']
                       if 'duration_beat' in window.dtype.names else 1.0)
                if step in step_to_tpc:
                    tpc = step_to_tpc[step] + 7 * alter
                    idx = tpc + CENTER_IDX
                    if 0 <= idx < 35:
                        dist[idx] += dur
            results.append({'piece_id': pid, 'distribution': dist})
        except Exception:
            continue
    print(f"Extracted {len(results)} windowed pieces (35-D).")
    out = OUTPUT_DIR / "pop_windowed_16bar_35d.npz"
    np.savez(out,
             piece_ids=[r['piece_id'] for r in results],
             distributions=np.array([r['distribution'] for r in results]))
    print(f"Saved to {out}")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified data extraction pipeline")
    parser.add_argument(
        'mode', choices=['classical', 'pop35d', 'pop12d', 'windowed', 'all'],
        help="Extraction mode")
    args = parser.parse_args()

    if args.mode in ('classical', 'all'):
        extract_classical_35d()
    if args.mode in ('pop35d', 'all'):
        extract_pop_35d()
    if args.mode in ('pop12d', 'all'):
        extract_pop_12d()
    if args.mode in ('windowed', 'all'):
        extract_windowed_classical()
        extract_windowed_pop()
