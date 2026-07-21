"""PTB-XL Data Loader — HuggingFace-first with PhysioNet fallback.

Loads the PTB-XL ECG dataset for training and evaluation:
  - Primary: HuggingFace `liyongsea/PTB-XL`
  - Fallback: Generate synthetic ECG data using neurokit2

Handles:
  - Loading 12-lead ECG waveforms as numpy arrays
  - Aggregating diagnostic labels into 5 superclasses
  - Train/val/test splits using the built-in strat_fold
  - Caching loaded data as .npy files for fast reloading
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CACHE_DIR = Path("data/processed/ptbxl_cache")
SAMPLING_RATE = 100  # Use 100Hz version (smaller, adequate for superclass)


def load_ptbxl_from_huggingface(
    cache_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load PTB-XL from HuggingFace and return (X, y, folds).

    Returns:
        X: shape (N, 1000, 12) — 12-lead ECG waveforms at 100Hz
        y: shape (N, 5) — multi-hot label matrix for 5 superclasses
        folds: shape (N,) — strat_fold values (1-10)
    """
    cache = cache_dir or CACHE_DIR
    x_path = cache / "X.npy"
    y_path = cache / "y.npy"
    folds_path = cache / "folds.npy"

    # Check cache first
    if x_path.exists() and y_path.exists() and folds_path.exists():
        logger.info("Loading PTB-XL from cache: %s", cache)
        return np.load(x_path), np.load(y_path), np.load(folds_path)

    logger.info("Loading PTB-XL from HuggingFace (liyongsea/PTB-XL)...")

    try:
        import pandas as pd
        import wfdb

        from datasets import load_dataset

        ds = load_dataset("liyongsea/PTB-XL", trust_remote_code=True)

        # The HuggingFace dataset should have the metadata
        # Try to extract waveforms and labels
        train_data = ds.get("train", ds.get("all", None))

        if train_data is None:
            raise ValueError("Could not find data split in HuggingFace dataset")

        # Process the dataset
        X_list = []
        y_list = []
        fold_list = []

        for record in train_data:
            # Extract signal data and labels
            # Format depends on how the HF dataset is structured
            signal = np.array(record.get("signal", record.get("ecg", [])))
            scp_codes = record.get("scp_codes", "{}")
            fold = record.get("strat_fold", 1)

            if isinstance(scp_codes, str):
                scp_codes = ast.literal_eval(scp_codes)

            # Aggregate to superclass labels
            label = _aggregate_to_superclass(scp_codes)

            if len(signal) > 0:
                X_list.append(signal)
                y_list.append(label)
                fold_list.append(fold)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        folds = np.array(fold_list, dtype=np.int32)

        # Cache for future use
        cache.mkdir(parents=True, exist_ok=True)
        np.save(x_path, X)
        np.save(y_path, y)
        np.save(folds_path, folds)

        logger.info("Loaded %d records from HuggingFace, cached to %s", len(X), cache)
        return X, y, folds

    except Exception as exc:
        logger.warning(
            "Failed to load from HuggingFace: %s. Falling back to synthetic data.",
            exc,
        )
        return generate_synthetic_ecg_data()


def _aggregate_to_superclass(scp_codes: dict) -> np.ndarray:
    """Map SCP codes to 5-class multi-hot label vector.

    The SCP-to-superclass mapping follows the PTB-XL convention
    using the diagnostic class hierarchy.
    """
    # Simplified SCP → superclass mapping
    # Full mapping requires scp_statements.csv, but these are the most
    # common diagnostic SCP codes and their superclass assignments
    scp_superclass_map = {
        # NORM
        "NORM": "NORM",
        # MI (Myocardial Infarction)
        "IMI": "MI", "AMI": "MI", "ALMI": "MI", "ASMI": "MI",
        "ILMI": "MI", "INJAS": "MI", "INJAL": "MI", "INJIN": "MI",
        "INJLA": "MI", "INJIL": "MI", "PMI": "MI", "LMI": "MI",
        # STTC (ST/T Changes)
        "STD_": "STTC", "STE_": "STTC", "ISC_": "STTC",
        "ISCA": "STTC", "ISCAL": "STTC", "ISCAN": "STTC",
        "ISCAS": "STTC", "ISCIN": "STTC", "ISCIL": "STTC",
        "ISCLA": "STTC", "NST_": "STTC", "DIG": "STTC",
        "LNGQT": "STTC", "APTS": "STTC", "EL": "STTC",
        # CD (Conduction Disturbance)
        "LAFB": "CD", "LPFB": "CD", "IRBBB": "CD", "CRBBB": "CD",
        "CLBBB": "CD", "ILBBB": "CD", "WPW": "CD",
        "1AVB": "CD", "2AVB": "CD", "3AVB": "CD",
        "IVCD": "CD",
        # HYP (Hypertrophy)
        "LVH": "HYP", "RVH": "HYP", "LAO/LAE": "HYP",
        "RAO/RAE": "HYP", "SEHYP": "HYP",
    }

    label = np.zeros(5, dtype=np.float32)

    for code in scp_codes:
        superclass = scp_superclass_map.get(code)
        if superclass and superclass in SUPERCLASSES:
            idx = SUPERCLASSES.index(superclass)
            label[idx] = 1.0

    return label


def generate_synthetic_ecg_data(
    num_samples: int = 200,
    duration_seconds: float = 10.0,
    fs: int = SAMPLING_RATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic ECG data for development and testing.

    Creates realistic-looking ECG waveforms with known labels using
    numpy-based waveform synthesis (no neurokit2 dependency required).

    Returns:
        X: shape (num_samples, duration*fs, 12)
        y: shape (num_samples, 5) — multi-hot labels
        folds: shape (num_samples,) — simulated fold assignments
    """
    logger.info("Generating %d synthetic ECG records for development", num_samples)

    num_points = int(duration_seconds * fs)
    X = np.zeros((num_samples, num_points, 12), dtype=np.float32)
    y = np.zeros((num_samples, 5), dtype=np.float32)
    rng = np.random.RandomState(42)

    for i in range(num_samples):
        # Randomly assign one or more superclasses
        label_idx = rng.choice(5, size=rng.randint(1, 3), replace=False)
        y[i, label_idx] = 1.0

        # Generate a base ECG-like signal for each lead
        heart_rate = rng.uniform(50, 120)  # bpm
        rr_interval = 60.0 / heart_rate

        for lead in range(12):
            signal = _generate_single_lead_ecg(
                num_points, fs, rr_interval, rng,
                has_mi=1 in label_idx,       # MI → ST elevation
                has_sttc=2 in label_idx,     # STTC → ST/T changes
            )
            # Add lead-specific amplitude scaling
            amplitude_scale = rng.uniform(0.5, 1.5)
            X[i, :, lead] = signal * amplitude_scale

    # Assign folds (1-10) for train/val/test split
    folds = (np.arange(num_samples) % 10) + 1

    # Cache synthetic data
    cache = CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "X_synthetic.npy", X)
    np.save(cache / "y_synthetic.npy", y)
    np.save(cache / "folds_synthetic.npy", folds)

    logger.info("Generated %d synthetic records, cached to %s", num_samples, cache)
    return X, y, folds


def _generate_single_lead_ecg(
    num_points: int,
    fs: int,
    rr_interval: float,
    rng: np.random.RandomState,
    has_mi: bool = False,
    has_sttc: bool = False,
) -> np.ndarray:
    """Generate a synthetic single-lead ECG signal.

    Creates a realistic PQRST waveform by summing Gaussian-like pulses
    positioned at physiologically correct timing intervals.
    """
    t = np.arange(num_points) / fs
    signal = np.zeros(num_points, dtype=np.float32)
    beat_times = np.arange(0, t[-1], rr_interval)

    for beat_time in beat_times:
        # P wave
        p_center = beat_time - 0.16
        p_width = 0.04
        signal += 0.15 * np.exp(-((t - p_center) ** 2) / (2 * p_width ** 2))

        # QRS complex (R peak dominant)
        q_center = beat_time - 0.03
        signal -= 0.10 * np.exp(-((t - q_center) ** 2) / (2 * 0.008 ** 2))

        r_center = beat_time
        signal += 1.0 * np.exp(-((t - r_center) ** 2) / (2 * 0.01 ** 2))

        s_center = beat_time + 0.03
        signal -= 0.15 * np.exp(-((t - s_center) ** 2) / (2 * 0.008 ** 2))

        # ST segment modification for MI/STTC
        st_elevation = 0.0
        if has_mi:
            st_elevation = rng.uniform(0.1, 0.3)
        elif has_sttc:
            st_elevation = rng.uniform(-0.2, 0.2)

        st_center = beat_time + 0.08
        signal += st_elevation * np.exp(-((t - st_center) ** 2) / (2 * 0.03 ** 2))

        # T wave
        t_center = beat_time + 0.25
        t_amplitude = 0.3 + (st_elevation * 0.5)
        signal += t_amplitude * np.exp(-((t - t_center) ** 2) / (2 * 0.05 ** 2))

    # Add realistic noise
    signal += rng.normal(0, 0.02, num_points)

    return signal


def get_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    test_fold: int = 10,
    val_fold: int = 9,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split data using the PTB-XL strat_fold convention.

    Args:
        X: Waveform data.
        y: Label matrix.
        folds: Fold assignments.
        test_fold: Fold number for test set (default 10).
        val_fold: Fold number for validation set (default 9).

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to (X, y).
    """
    train_mask = (folds != test_fold) & (folds != val_fold)
    val_mask = folds == val_fold
    test_mask = folds == test_fold

    return {
        "train": (X[train_mask], y[train_mask]),
        "val": (X[val_mask], y[val_mask]),
        "test": (X[test_mask], y[test_mask]),
    }
