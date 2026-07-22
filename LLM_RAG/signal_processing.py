"""
04_signal_processing.py
EEG + ECG preprocessing utilities for SpikeRecorder data.
Fully compatible with NumPy 1.x and NumPy 2.x.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch
from pathlib import Path
from config import EEG_FS, ECG_FS, BANDS, ECG_BAND

# NumPy 2.x removed np.trapz → use numpy.trapezoid
try:
    from numpy import trapezoid as integrate_trapz
except ImportError:
    # fallback for NumPy 1.x
    from numpy import trapz as integrate_trapz


# ------------------------------------------------------------
# EEG preprocessing
# ------------------------------------------------------------
def preprocess_eeg(x, fs=EEG_FS):
    x = x - x.mean()

    sos = butter(4, [1, 45], "band", fs=fs, output="sos")
    x = sosfiltfilt(sos, x)

    b, a = iirnotch(50, 30, fs)
    x = filtfilt(b, a, x)

    return x


# ------------------------------------------------------------
# ECG preprocessing
# ------------------------------------------------------------
def preprocess_ecg(x, fs=ECG_FS):
    x = x - x.mean()
    sos = butter(3, ECG_BAND, "band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


# ------------------------------------------------------------
# Bandpower (Welch + trapezoid integration)
# ------------------------------------------------------------
def bandpowers(x, fs=EEG_FS):
    """
    Compute absolute + relative bandpowers.
    Uses numpy.trapezoid for NumPy 2.x compatibility.
    """

    nper = int(min(len(x), fs * 2))
    if nper < fs // 2:
        return {k: np.nan for k in BANDS} | {k + "_rel": np.nan for k in BANDS}

    f, p = welch(x, fs, nperseg=nper)

    # Total power in 1–45 Hz
    total_mask = (f >= 1) & (f < 45)
    total_power = integrate_trapz(p[total_mask], f[total_mask])

    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (f >= lo) & (f < hi)
        bp = integrate_trapz(p[mask], f[mask])
        out[name] = bp
        out[name + "_rel"] = bp / total_power if total_power > 0 else np.nan

    return out

# ------------------------------------------------------------
# Alpha envelope (Hilbert transform)
# ------------------------------------------------------------
from scipy.signal import hilbert

def alpha_envelope(x, fs=EEG_FS):
    if x is None or len(x) == 0:
        return np.array([])
    low = 8 / (fs / 2)
    high = 12 / (fs / 2)
    b, a = butter(4, [low, high], btype="band")
    x_alpha = filtfilt(b, a, x)
    analytic = hilbert(x_alpha)
    return np.abs(analytic)


# ------------------------------------------------------------
# ECG bandpass (used by Streamlit app)
# ------------------------------------------------------------
def bandpass_ecg(x, fs=ECG_FS):
    if x is None or len(x) == 0:
        return np.array([])
    x = x - x.mean()
    sos = butter(3, ECG_BAND, "band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

# ------------------------------------------------------------
# R-peak detection (simple Pan–Tompkins style)
# ------------------------------------------------------------
def detect_rpeaks(ecg, fs=ECG_FS):
    """
    Lightweight R-peak detector compatible with SpikeRecorder ECG.
    Steps:
    1. Bandpass (already done in preprocess_ecg or bandpass_ecg)
    2. Differentiate
    3. Square
    4. Moving window integration
    5. Threshold + peak picking
    """
    if ecg is None or len(ecg) == 0:
        return np.array([])

    # Differentiate
    diff = np.diff(ecg)

    # Square
    sq = diff**2

    # Moving window integration (~150 ms)
    win = int(0.15 * fs)
    if win < 1:
        win = 1
    mwa = np.convolve(sq, np.ones(win)/win, mode="same")

    # Threshold
    thr = np.mean(mwa) + 0.5*np.std(mwa)

    # Peak picking
    peaks = np.where(mwa > thr)[0]

    # Remove peaks too close together (<200 ms)
    cleaned = []
    last = -999
    for p in peaks:
        if p - last > int(0.2 * fs):
            cleaned.append(p)
            last = p

    return np.array(cleaned)

# ------------------------------------------------------------
# RR interval cleaning
# ------------------------------------------------------------
def clean_rr(peaks, fs=ECG_FS):
    """
    Convert R-peaks → RR intervals (ms), remove physiologically impossible values.
    """
    if peaks is None or len(peaks) < 2:
        return np.array([])

    rr = np.diff(peaks) * (1000.0 / fs)  # ms

    # Remove RR < 300 ms or > 2000 ms
    rr = rr[(rr > 300) & (rr < 2000)]

    return rr


# ------------------------------------------------------------
# Find recording for subject + condition (Streamlit only)
# ------------------------------------------------------------
def find_rec(subject, condition):
    """
    Load raw EEG + ECG for Streamlit dashboard.
    Does NOT affect pipeline. Uses artifacts/raw_eeg and raw_ecg.
    """

    base = Path(__file__).resolve().parents[1] / "artifacts"

    eeg_path = base / "raw_eeg" / f"{subject}_{condition}.npy"
    ecg_path = base / "raw_ecg" / f"{subject}_{condition}.npy"

    if not eeg_path.exists():
        raise FileNotFoundError(f"Missing EEG file: {eeg_path}")

    eeg_raw = np.load(eeg_path)
    ecg_raw = np.load(ecg_path) if ecg_path.exists() else None

    return EEG_FS, eeg_raw, ECG_FS, ecg_raw, None
