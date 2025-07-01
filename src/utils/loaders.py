import numpy as np
import pandas as pd
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
TDM_RESULTS_DIR = RESULTS_DIR / "tdm_analysis"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_lmd_distributions():
    """Load Lakh MIDI Dataset (12-D) distributions."""
    data_path = RESULTS_DIR / "data" / "lmd_pitch_class_distributions.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"LMD data not found at {data_path}")
    return np.load(data_path, allow_pickle=True)

def load_lmd_35d_distributions():
    """Load Lakh MIDI Dataset (35-D Spelled) distributions."""
    data_path = RESULTS_DIR / "data" / "lmd_pitch_class_distributions_35d_partitura.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"LMD 35-D data not found at {data_path}")
    return np.load(data_path, allow_pickle=True)

def load_classical_35d_distributions():
    """Load Distant Listening Corpus 35-D distributions."""
    # FIX: Data is in RESULTS_DIR/data, not DATA_DIR
    path = RESULTS_DIR / "data" / "classical_pitch_class_distributions.npz"
    if not path.exists():
        # Fallback to DATA_DIR just in case
        path = DATA_DIR / "classical_pitch_class_distributions.npz"
        if not path.exists():
            raise FileNotFoundError(f"Classical 35-D data not found at {path}")
    
    data = np.load(path, allow_pickle=True)
    result = {
        'distributions': data['distributions'],
        'piece_ids': data['piece_ids']
    }
    if 'durations' in data:
        result['durations'] = data['durations']
    return result

def load_classical_metadata():
    """Load the DCML corpus metadata."""
    meta_path = DATA_DIR / "distant_listening_corpus" / "distant_listening_corpus.metadata.tsv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}")
    return pd.read_csv(meta_path, sep='\t')

def key_str_to_tpc(key_str):
    """Convert key string to TPC index (0=C, 1=G, etc.)."""
    if not isinstance(key_str, str) or pd.isna(key_str):
        return None
    
    key_str = key_str.strip()
    if not key_str:
        return None
        
    root_letter = key_str[0].upper()
    accidentals = key_str[1:]
    
    # Base TPC (Circle of Fifths position: F=-1, C=0, G=1...)
    # F C G D A E B
    # -1 0 1 2 3 4 5
    base_map = {
        'F': -1, 'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5
    }
    
    if root_letter not in base_map:
        return None 
        
    tpc = base_map[root_letter]
    for char in accidentals:
        if char == '#': tpc += 7
        elif char == 'b': tpc -= 7
        
    return tpc

def get_annotated_tonal_centers(piece_ids):
    """Get annotated tonal centers for a list of piece IDs."""
    try:
        meta_df = load_classical_metadata()
    except:
        print("Warning: Could not load metadata.")
        return [None] * len(piece_ids)
        
    piece_to_key = dict(zip(meta_df['piece'], meta_df['annotated_key']))
    
    centers = []
    center_idx = 17 # C = 0 -> Index 17 in 35-D array
    
    for pid in piece_ids:
        if pid in piece_to_key:
            tpc = key_str_to_tpc(piece_to_key[pid])
            if tpc is not None:
                # Convert TPC (-1, 0, 1...) to array index (16, 17, 18...)
                # Note: TDM model usually expects index in the distribution array
                centers.append(tpc + center_idx)
            else:
                centers.append(None)
        else:
            centers.append(None)
            
    return centers

def project_to_12d(dist_35d):
    """Project 35-D Line of Fifths distribution to 12-D Pitch Classes."""
    dist_12d = np.zeros(12)
    center_idx = 17  # C
    
    for i, weight in enumerate(dist_35d):
        if weight > 0:
            # Map LoF index to Pitch Class (C=0)
            # Each step on LoF is +7 semitones (P5)
            # C=0, G=7, D=2, A=9, E=4, B=11, F#=6, C#=1...
            # Wait, standard Circle of Fifths order for TDM is usually:
            # 0=C, 1=G, 2=D... (steps of 1 fifth)
            # My TDM implementation assumes steps of 1.
            # So I should map LoF index directly to CoF index modulo 12.
            # LoF: ... F(-1) C(0) G(1) D(2) ...
            # CoF: ... 11    0    1    2 ...
            # So it's just (i - center_idx) % 12.
            
            cof_idx = (i - center_idx) % 12
            dist_12d[cof_idx] += weight
            
    return dist_12d

def load_classical_12d_distributions():
    """Load Distant Listening Corpus and project to 12-D."""
    data = load_classical_35d_distributions()
    distributions_35d = data['distributions']
    
    distributions_12d = np.array([project_to_12d(d) for d in distributions_35d])
    
    result = {
        'distributions': distributions_12d,
        'piece_ids': data['piece_ids']
    }
    if 'durations' in data:
        result['durations'] = data['durations']
    return result

def load_improved_tdm_results():
    """Load the TDM analysis results as separate DataFrames."""
    results_dir = TDM_RESULTS_DIR / "improved_tdm_results"
    
    classical_path = results_dir / "classical_tdm_improved.csv"
    pop_path = results_dir / "pop_tdm_improved.csv"
    
    if not classical_path.exists() or not pop_path.exists():
        print("Warning: TDM results not found. Please run run_analysis.py first.")
        return pd.DataFrame(), pd.DataFrame()
        
    classical_df = pd.read_csv(classical_path)
    pop_df = pd.read_csv(pop_path)
    
    return classical_df, pop_df


def load_all_tdm_results():
    """Load all TDM results as a single DataFrame with 'corpus' column.

    Tries the pre-built all_tdm_results.csv first (created by tdm_fitting.py).
    Falls back to merging classical_tdm_improved.csv + pop_tdm_improved.csv.
    """
    results_dir = TDM_RESULTS_DIR / "improved_tdm_results"
    all_path = results_dir / "all_tdm_results.csv"

    if all_path.exists():
        return pd.read_csv(all_path)

    # Merge the two tracked CSVs
    classical_path = results_dir / "classical_tdm_improved.csv"
    pop_path = results_dir / "pop_tdm_improved.csv"

    if not classical_path.exists() or not pop_path.exists():
        raise FileNotFoundError(
            f"TDM results not found. Run: python -m src.experiments.tdm_fitting")

    c_df = pd.read_csv(classical_path)
    c_df['corpus'] = 'Classical'
    p_df = pd.read_csv(pop_path)
    p_df['corpus'] = 'Pop'

    return pd.concat([c_df, p_df], ignore_index=True)

def load_35d_distributions():
    """Load the 35-D distributions."""
    dcml_path = DATA_DIR / "dcml_pitch_class_distributions.npz"
    bimmuda_path = DATA_DIR / "bimmuda_pitch_class_distributions_35d.npz"
    
    if not dcml_path.exists():
        raise FileNotFoundError(f"Classical data not found at {dcml_path}")
    if not bimmuda_path.exists():
        raise FileNotFoundError(f"Pop data not found at {bimmuda_path}")
        
    dcml_data = np.load(dcml_path, allow_pickle=True)
    bimmuda_data = np.load(bimmuda_path, allow_pickle=True)
    
    return dcml_data, bimmuda_data


# --- Krumhansl-Schmuckler Key Estimation ---

# Krumhansl-Kessler key profiles (from Krumhansl 1990)
# These are for 12-D chromatic pitch classes (C, C#, D, ..., B)
KS_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def _project_35d_to_12d_chromatic(dist_35d):
    """
    Project 35-D Line-of-Fifths distribution to 12-D chromatic pitch classes.
    
    LoF index 17 = C (pitch class 0)
    Each step on LoF = +7 semitones (perfect fifth)
    
    Returns: 12-D array where index 0=C, 1=C#, 2=D, ..., 11=B
    """
    dist_12d = np.zeros(12)
    center_idx = 17  # C on LoF
    
    for i, weight in enumerate(dist_35d):
        if weight > 0:
            # LoF position relative to C
            lof_offset = i - center_idx
            # Each step on LoF = +7 semitones
            semitone = (lof_offset * 7) % 12
            dist_12d[semitone] += weight
    
    return dist_12d


def estimate_key_ks(dist_35d):
    """
    Estimate key using Krumhansl-Schmuckler algorithm.
    
    Parameters:
    -----------
    dist_35d : np.array, shape (35,)
        Pitch distribution on Line-of-Fifths
    
    Returns:
    --------
    dict with:
        'key_index_35d': int (0-34), position on LoF
        'key_name': str (e.g., 'C', 'Am')
        'mode': str ('major' or 'minor')
        'correlation': float, correlation with best-matching profile
    """
    if dist_35d.sum() == 0:
        return {'key_index_35d': None, 'key_name': None, 'mode': None, 'correlation': 0}
    
    # Project to 12-D chromatic
    dist_12d = _project_35d_to_12d_chromatic(dist_35d)
    
    # Normalize
    if dist_12d.sum() > 0:
        dist_12d = dist_12d / dist_12d.sum()
    else:
        return {'key_index_35d': None, 'key_name': None, 'mode': None, 'correlation': 0}
    
    best_corr = -2
    best_key = 0
    best_mode = 'major'
    
    # Test all 24 keys (12 major + 12 minor)
    for key in range(12):
        # Rotate profile to match key
        major_profile = np.roll(KS_MAJOR_PROFILE, key)
        minor_profile = np.roll(KS_MINOR_PROFILE, key)
        
        # Pearson correlation
        major_corr = np.corrcoef(dist_12d, major_profile)[0, 1]
        minor_corr = np.corrcoef(dist_12d, minor_profile)[0, 1]
        
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = key
            best_mode = 'major'
        
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = key
            best_mode = 'minor'
    
    # Map chromatic key to LoF index
    # Chromatic: 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
    # LoF: Each chromatic pitch has a position on LoF
    # C=0, G=1, D=2, A=3, E=4, B=5, F#=6, C#=7, G#=8, D#/Eb=9, Bb=-2, F=-1
    # Mapping chromatic to CoF position:
    chromatic_to_cof = {
        0: 0,   # C
        1: 7,   # C# (7 fifths up from C)
        2: 2,   # D
        3: 9,   # D#/Eb (as Eb: -3, as D#: 9)
        4: 4,   # E
        5: -1,  # F
        6: 6,   # F#
        7: 1,   # G
        8: 8,   # G# (as Ab: -4, as G#: 8)
        9: 3,   # A
        10: -2, # Bb
        11: 5   # B
    }
    
    cof_position = chromatic_to_cof[best_key]
    lof_index = cof_position + 17  # Center (C) is at index 17
    
    # Clamp to valid range
    lof_index = max(0, min(34, lof_index))
    
    # Key name
    key_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    key_name = key_names[best_key]
    if best_mode == 'minor':
        key_name = key_name.lower() + 'm'
    
    return {
        'key_index_35d': lof_index,
        'key_name': key_name,
        'mode': best_mode,
        'correlation': best_corr
    }


def get_ks_tonal_centers(distributions):
    """
    Get Krumhansl-Schmuckler estimated tonal centers for a list of distributions.
    
    Parameters:
    -----------
    distributions : np.array, shape (n_pieces, 35)
        Pitch distributions on Line-of-Fifths
    
    Returns:
    --------
    list of int or None: Tonal center indices (0-34) for each piece
    """
    centers = []
    for dist in distributions:
        result = estimate_key_ks(dist)
        centers.append(result['key_index_35d'])
    return centers

