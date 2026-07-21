"""ECG DSP Feature Extraction Pipeline.

Extracts clinical features from raw 12-lead ECG waveforms using
digital signal processing:

  1. Bandpass filter (0.5–45 Hz) — removes baseline wander + HF noise
  2. R-peak detection (NeuroKit2 / scipy fallback)
  3. Wave delineation — P, QRS, T onsets/offsets
  4. Feature computation — HR, QRS duration, QT/QTc, ST deviation, etc.

This module is pure DSP — no ML model involved. The extracted features
are fed to the LLM as structured numerical context (not raw waveforms).

Lead naming convention (standard 12-lead ECG):
  Limb: I, II, III, aVR, aVL, aVF
  Precordial: V1, V2, V3, V4, V5, V6
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy.signal import butter, filtfilt, find_peaks

logger = logging.getLogger(__name__)

# Standard 12-lead names in the order used by PTB-XL
LEAD_NAMES_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

# Lead II is the primary analysis lead (best P-wave and R-wave morphology)
PRIMARY_LEAD_INDEX = 1  # Lead II


# ── Output Models ────────────────────────────────────────────────────────────

class ECGFeatures(BaseModel):
    """Structured features extracted from a single ECG lead."""

    lead_name: str = Field(
        ..., description="Name of the ECG lead (e.g. 'II', 'V1')"
    )
    sampling_rate: int = Field(
        ..., description="Sampling rate in Hz"
    )
    heart_rate_bpm: float = Field(
        ..., description="Mean heart rate derived from RR intervals"
    )
    qrs_duration_ms: Optional[float] = Field(
        default=None, description="QRS complex duration in milliseconds"
    )
    qt_interval_ms: Optional[float] = Field(
        default=None, description="QT interval in milliseconds"
    )
    qtc_bazett_ms: Optional[float] = Field(
        default=None, description="Corrected QT interval (Bazett's formula)"
    )
    pr_interval_ms: Optional[float] = Field(
        default=None, description="PR interval in milliseconds"
    )
    st_deviation_mv: Optional[float] = Field(
        default=None,
        description="ST segment deviation in mV (positive=elevation, negative=depression)",
    )
    p_wave_duration_ms: Optional[float] = Field(
        default=None, description="P-wave duration in milliseconds"
    )
    r_peak_count: int = Field(
        default=0, description="Number of R-peaks detected"
    )
    signal_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated signal quality (0=poor, 1=excellent)",
    )


class MultiLeadECGFeatures(BaseModel):
    """Aggregated features from all leads of a standard ECG."""

    per_lead: list[ECGFeatures] = Field(
        ..., description="Per-lead feature extraction results"
    )
    mean_heart_rate: float = Field(
        ..., description="Mean heart rate across analyzable leads"
    )
    heart_rate_min: float = Field(
        default=0.0, description="Minimum instantaneous heart rate"
    )
    heart_rate_max: float = Field(
        default=0.0, description="Maximum instantaneous heart rate"
    )
    rhythm_regular: bool = Field(
        default=True,
        description="True if RR interval variability is below threshold",
    )
    num_leads_analyzed: int = Field(
        ..., description="Number of leads that produced usable features"
    )
    overall_signal_quality: float = Field(
        default=0.0, description="Mean signal quality across all leads"
    )


# ── DSP Functions ────────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: int = 100,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to an ECG signal.

    Removes baseline wander (<0.5 Hz) and high-frequency noise (>45 Hz)
    while preserving the PQRST morphology.

    Args:
        signal: 1D numpy array of ECG samples.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal as 1D numpy array.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # Clamp to valid range
    low = max(low, 0.001)
    high = min(high, 0.999)

    if low >= high:
        logger.warning("Invalid filter cutoffs (low=%f, high=%f), returning unfiltered", low, high)
        return signal

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, padlen=min(len(signal) - 1, 3 * max(len(a), len(b))))


def detect_r_peaks(
    signal: np.ndarray,
    fs: int = 100,
    min_distance_ms: float = 300.0,
) -> np.ndarray:
    """Detect R-peaks in a filtered ECG signal.

    Uses scipy's find_peaks with adaptive height threshold.
    This is a simplified but robust approach:
      1. Compute the signal's moving RMS for adaptive thresholding
      2. Find peaks above threshold with minimum distance constraint

    Args:
        signal: Filtered 1D ECG signal.
        fs: Sampling rate in Hz.
        min_distance_ms: Minimum distance between R-peaks in ms.

    Returns:
        Array of R-peak sample indices.
    """
    min_distance_samples = int(min_distance_ms * fs / 1000)

    # Adaptive threshold: peaks must be above 0.5× the signal's
    # 90th percentile absolute amplitude
    abs_signal = np.abs(signal)
    threshold = 0.5 * np.percentile(abs_signal, 90)

    peaks, properties = find_peaks(
        signal,
        height=threshold,
        distance=max(min_distance_samples, 1),
        prominence=threshold * 0.3,
    )

    return peaks


def estimate_signal_quality(signal: np.ndarray, fs: int = 100) -> float:
    """Estimate ECG signal quality using signal-to-noise ratio heuristics.

    Quality score based on:
      1. Signal amplitude range (too flat or clipped = bad)
      2. Baseline stability (excessive wander = bad)
      3. High-frequency noise level

    Returns:
        Quality score between 0.0 (unusable) and 1.0 (excellent).
    """
    if len(signal) < fs:  # less than 1 second
        return 0.0

    # Check for flat signals
    amplitude_range = np.ptp(signal)
    if amplitude_range < 0.01:
        return 0.0

    # Check for clipping (values at min/max for extended periods)
    clip_fraction = np.mean(
        (np.abs(signal - np.max(signal)) < 0.01)
        | (np.abs(signal - np.min(signal)) < 0.01)
    )
    clip_score = max(0, 1.0 - clip_fraction * 10)

    # High-frequency noise estimate (energy above 45Hz vs total)
    from scipy.signal import welch
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs * 2))
    total_power = np.sum(psd)
    if total_power < 1e-12:
        return 0.0

    signal_power = np.sum(psd[freqs <= 45])
    noise_ratio = signal_power / total_power
    noise_score = min(noise_ratio, 1.0)

    # Combined score
    quality = 0.5 * noise_score + 0.3 * clip_score + 0.2 * min(amplitude_range / 2.0, 1.0)
    return float(np.clip(quality, 0.0, 1.0))


def compute_rr_intervals(r_peaks: np.ndarray, fs: int = 100) -> np.ndarray:
    """Compute RR intervals in seconds from R-peak indices."""
    if len(r_peaks) < 2:
        return np.array([])
    return np.diff(r_peaks) / fs


def compute_heart_rate(rr_intervals: np.ndarray) -> tuple[float, float, float]:
    """Compute mean, min, max heart rate from RR intervals.

    Returns:
        Tuple of (mean_hr, min_hr, max_hr) in bpm.
    """
    if len(rr_intervals) == 0:
        return 0.0, 0.0, 0.0

    # Filter out physiologically implausible RR intervals
    valid_rr = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
    if len(valid_rr) == 0:
        return 0.0, 0.0, 0.0

    hr = 60.0 / valid_rr
    return float(np.mean(hr)), float(np.min(hr)), float(np.max(hr))


def check_rhythm_regularity(
    rr_intervals: np.ndarray, threshold: float = 0.15
) -> bool:
    """Check if the rhythm is regular based on RR interval variability.

    Args:
        rr_intervals: Array of RR intervals in seconds.
        threshold: Coefficient of variation threshold for irregularity.

    Returns:
        True if rhythm is regular, False otherwise.
    """
    if len(rr_intervals) < 3:
        return True  # not enough data to determine

    valid_rr = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
    if len(valid_rr) < 3:
        return True

    cv = np.std(valid_rr) / np.mean(valid_rr)
    return float(cv) < threshold


def estimate_qrs_duration(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 100,
) -> Optional[float]:
    """Estimate QRS duration by measuring the width of the R-peak complex.

    Uses a simple approach: measure the duration where the signal's
    derivative is above a threshold around each R-peak.

    Returns:
        Mean QRS duration in milliseconds, or None if estimation fails.
    """
    if len(r_peaks) < 2:
        return None

    derivative = np.gradient(signal)
    abs_derivative = np.abs(derivative)

    qrs_durations = []
    for peak in r_peaks:
        # Search window: ±150ms around R-peak
        window_samples = int(0.15 * fs)
        start = max(0, peak - window_samples)
        end = min(len(signal), peak + window_samples)

        if end - start < 3:
            continue

        window_deriv = abs_derivative[start:end]
        threshold = 0.3 * np.max(window_deriv)

        # Find where derivative is above threshold (QRS complex region)
        above_threshold = window_deriv > threshold
        if not np.any(above_threshold):
            continue

        indices = np.where(above_threshold)[0]
        qrs_samples = indices[-1] - indices[0] + 1
        qrs_ms = (qrs_samples / fs) * 1000

        # Physiological plausibility: QRS should be 60-200ms
        if 40 <= qrs_ms <= 250:
            qrs_durations.append(qrs_ms)

    if not qrs_durations:
        return None

    return float(np.median(qrs_durations))


def estimate_st_deviation(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 100,
) -> Optional[float]:
    """Estimate ST segment deviation relative to the isoelectric baseline.

    Measures the voltage at the J-point (end of QRS, ~80ms after R-peak)
    relative to the PR segment baseline (~40ms before the R-peak).

    Returns:
        ST deviation in mV (positive=elevation, negative=depression).
        None if estimation fails.
    """
    if len(r_peaks) < 2:
        return None

    st_deviations = []
    for peak in r_peaks:
        # J-point: approximately 80ms after R-peak
        j_point = peak + int(0.08 * fs)
        # Baseline: approximately 40ms before R-peak (PR segment)
        baseline_point = peak - int(0.04 * fs)

        if baseline_point < 0 or j_point >= len(signal):
            continue

        # Average over a small window for stability
        j_window = int(0.02 * fs)  # 20ms window
        b_window = int(0.02 * fs)

        j_start = max(0, j_point - j_window // 2)
        j_end = min(len(signal), j_point + j_window // 2 + 1)
        b_start = max(0, baseline_point - b_window // 2)
        b_end = min(len(signal), baseline_point + b_window // 2 + 1)

        if j_end <= j_start or b_end <= b_start:
            continue

        j_voltage = np.mean(signal[j_start:j_end])
        baseline_voltage = np.mean(signal[b_start:b_end])

        st_dev = j_voltage - baseline_voltage
        st_deviations.append(st_dev)

    if not st_deviations:
        return None

    return float(np.median(st_deviations))


def estimate_qt_interval(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 100,
) -> tuple[Optional[float], Optional[float]]:
    """Estimate QT interval and corrected QTc (Bazett's formula).

    QT interval = Q onset to T offset.
    Simplified approach: measure from QRS onset to the end of the T-wave.

    Returns:
        Tuple of (qt_ms, qtc_ms). Either may be None if estimation fails.
    """
    if len(r_peaks) < 2:
        return None, None

    rr_intervals = compute_rr_intervals(r_peaks, fs)
    if len(rr_intervals) == 0:
        return None, None

    mean_rr = float(np.mean(rr_intervals))
    if mean_rr <= 0:
        return None, None

    qt_estimates = []
    for i, peak in enumerate(r_peaks[:-1]):
        # Q onset: ~40ms before R-peak
        q_onset = peak - int(0.04 * fs)
        if q_onset < 0:
            continue

        # T-wave end: search for the point where the signal returns to
        # baseline after the T-wave peak. The T-wave typically ends
        # at about 40-60% of the RR interval after the R-peak.
        search_end = peak + int(0.6 * mean_rr * fs)
        search_end = min(search_end, len(signal) - 1)
        t_search_start = peak + int(0.15 * fs)  # skip QRS complex

        if t_search_start >= search_end:
            continue

        t_segment = signal[t_search_start:search_end]
        if len(t_segment) < 3:
            continue

        # Find T-wave end: where the signal's derivative crosses zero
        # after the T-wave peak (approximation)
        derivative = np.gradient(t_segment)
        t_peak_idx = np.argmax(np.abs(t_segment))

        # Look for zero crossing after T-peak
        post_peak_deriv = derivative[t_peak_idx:]
        zero_crossings = np.where(np.diff(np.sign(post_peak_deriv)))[0]

        if len(zero_crossings) > 0:
            t_end_relative = t_peak_idx + zero_crossings[0]
            t_end_absolute = t_search_start + t_end_relative
        else:
            # Fallback: use a fixed fraction of RR interval
            t_end_absolute = peak + int(0.44 * mean_rr * fs)

        qt_samples = t_end_absolute - q_onset
        qt_ms = (qt_samples / fs) * 1000

        # Physiological plausibility: QT should be 300–600ms
        if 250 <= qt_ms <= 650:
            qt_estimates.append(qt_ms)

    if not qt_estimates:
        return None, None

    qt_ms = float(np.median(qt_estimates))

    # Bazett's correction: QTc = QT / sqrt(RR)
    qtc_ms = qt_ms / np.sqrt(mean_rr) if mean_rr > 0 else None

    return qt_ms, float(qtc_ms) if qtc_ms is not None else None


# ── Main Extraction Functions ────────────────────────────────────────────────

def extract_single_lead_features(
    signal: np.ndarray,
    lead_name: str,
    fs: int = 100,
) -> ECGFeatures:
    """Extract features from a single ECG lead.

    Args:
        signal: 1D numpy array of raw ECG samples.
        lead_name: Name of the ECG lead (e.g. "II", "V1").
        fs: Sampling rate in Hz.

    Returns:
        ECGFeatures with all extractable metrics.
    """
    # Step 1: Bandpass filter
    filtered = bandpass_filter(signal, fs=fs)

    # Step 2: Signal quality
    quality = estimate_signal_quality(filtered, fs=fs)

    # Step 3: R-peak detection
    r_peaks = detect_r_peaks(filtered, fs=fs)

    # Step 4: Compute intervals
    rr_intervals = compute_rr_intervals(r_peaks, fs=fs)
    mean_hr, _, _ = compute_heart_rate(rr_intervals)

    # Step 5: QRS duration
    qrs_duration = estimate_qrs_duration(filtered, r_peaks, fs=fs)

    # Step 6: ST deviation
    st_deviation = estimate_st_deviation(filtered, r_peaks, fs=fs)

    # Step 7: QT/QTc interval
    qt_ms, qtc_ms = estimate_qt_interval(filtered, r_peaks, fs=fs)

    return ECGFeatures(
        lead_name=lead_name,
        sampling_rate=fs,
        heart_rate_bpm=mean_hr,
        qrs_duration_ms=qrs_duration,
        qt_interval_ms=qt_ms,
        qtc_bazett_ms=qtc_ms,
        pr_interval_ms=None,  # requires P-wave detection (V2 upgrade)
        st_deviation_mv=st_deviation,
        p_wave_duration_ms=None,  # requires P-wave detection (V2 upgrade)
        r_peak_count=len(r_peaks),
        signal_quality=quality,
    )


def extract_multilead_features(
    waveform: np.ndarray,
    fs: int = 100,
    lead_names: list[str] | None = None,
) -> MultiLeadECGFeatures:
    """Extract features from a multi-lead ECG recording.

    Args:
        waveform: 2D numpy array of shape (num_samples, num_leads).
        fs: Sampling rate in Hz.
        lead_names: Optional list of lead names. Defaults to standard
                    12-lead names if waveform has 12 channels.

    Returns:
        MultiLeadECGFeatures with per-lead and aggregated metrics.
    """
    if waveform.ndim != 2:
        raise ValueError(f"Expected 2D waveform array, got shape {waveform.shape}")

    num_samples, num_leads = waveform.shape

    if lead_names is None:
        if num_leads == 12:
            lead_names = LEAD_NAMES_12
        else:
            lead_names = [f"Lead_{i}" for i in range(num_leads)]

    if len(lead_names) != num_leads:
        raise ValueError(
            f"lead_names length ({len(lead_names)}) != number of leads ({num_leads})"
        )

    logger.info(
        "Extracting features from %d-lead ECG (%d samples, %d Hz)",
        num_leads, num_samples, fs,
    )

    per_lead_features = []
    heart_rates = []

    for i, name in enumerate(lead_names):
        try:
            features = extract_single_lead_features(waveform[:, i], name, fs)
            per_lead_features.append(features)
            if features.heart_rate_bpm > 0:
                heart_rates.append(features.heart_rate_bpm)
        except Exception as exc:
            logger.warning("Failed to extract features for lead %s: %s", name, exc)
            # Add a minimal features entry for the failed lead
            per_lead_features.append(ECGFeatures(
                lead_name=name,
                sampling_rate=fs,
                heart_rate_bpm=0.0,
                r_peak_count=0,
                signal_quality=0.0,
            ))

    # Aggregate heart rate from primary lead (Lead II) if available,
    # otherwise use mean across leads
    primary_features = next(
        (f for f in per_lead_features if f.lead_name == "II" and f.heart_rate_bpm > 0),
        None,
    )

    if primary_features:
        mean_hr = primary_features.heart_rate_bpm
    elif heart_rates:
        mean_hr = float(np.mean(heart_rates))
    else:
        mean_hr = 0.0

    # Compute heart rate range from primary lead's RR intervals
    hr_min = min(heart_rates) if heart_rates else 0.0
    hr_max = max(heart_rates) if heart_rates else 0.0

    # Check rhythm regularity from primary lead
    primary_lead_idx = PRIMARY_LEAD_INDEX if num_leads > PRIMARY_LEAD_INDEX else 0
    filtered_primary = bandpass_filter(waveform[:, primary_lead_idx], fs=fs)
    primary_r_peaks = detect_r_peaks(filtered_primary, fs=fs)
    primary_rr = compute_rr_intervals(primary_r_peaks, fs=fs)
    rhythm_regular = check_rhythm_regularity(primary_rr)

    analyzable_leads = sum(1 for f in per_lead_features if f.signal_quality > 0.1)
    overall_quality = float(np.mean([f.signal_quality for f in per_lead_features]))

    return MultiLeadECGFeatures(
        per_lead=per_lead_features,
        mean_heart_rate=mean_hr,
        heart_rate_min=hr_min,
        heart_rate_max=hr_max,
        rhythm_regular=rhythm_regular,
        num_leads_analyzed=analyzable_leads,
        overall_signal_quality=overall_quality,
    )
