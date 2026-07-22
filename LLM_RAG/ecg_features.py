"""
06_ecg_features.py
Extract ECG features per epoch:
    - HR (autocorrelation)
    - RMSSD
    - SDNN
    - pNN50

Input:
    artifacts/raw_ecg/<subject>_<condition>.npy
    artifacts/ep.parquet

Output:
    artifacts/ecg_features.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, sosfiltfilt, decimate

from config import (
    ARTIFACTS_DIR,
    EP_PARQUET,
    ECG_FS,
    ECG_CH,
    log
)

ECG_FEATURES_PARQ = ARTIFACTS_DIR / "ecg_features.parquet"
RAW_ECG_DIR = ARTIFACTS_DIR / "raw_ecg"


# ------------------------------------------------------------
# ECG preprocessing (QRS band)
# ------------------------------------------------------------
def preprocess_ecg(x, fs=ECG_FS):
    sos = butter(3, [8, 20], "band", fs=fs, output="sos")
    return sosfiltfilt(sos, x - x.mean())


# ------------------------------------------------------------
# R-peak detection (Pan–Tompkins style)
# ------------------------------------------------------------
def detect_rpeaks(ecg, fs=ECG_FS):
    x = preprocess_ecg(ecg, fs)
    mwi = np.convolve(np.gradient(x)**2, np.ones(int(0.12*fs))/int(0.12*fs), mode="same")
    thr = 0.35 * np.median(mwi[mwi > np.percentile(mwi, 50)])
    pk, _ = find_peaks(mwi, height=thr, distance=int(0.45*fs))

    # refine peaks
    refined = []
    w = int(0.08 * fs)
    for p in pk:
        a, b = max(0, p-w), min(len(x), p+w)
        refined.append(a + np.argmax(np.abs(x[a:b])))

    return np.array(sorted(set(refined)))


# ------------------------------------------------------------
# HRV metrics
# ------------------------------------------------------------
def clean_rr(peaks, fs=ECG_FS):
    if len(peaks) < 5:
        return np.array([])

    rr = np.diff(peaks) / fs * 1000  # ms

    for _ in range(4):
        med = np.median(rr)
        keep = (rr > 0.7*med) & (rr < 1.5*med) & (rr > 300) & (rr < 1600)
        if keep.all():
            break
        rr = rr[keep]

    return rr


def compute_hrv(rr):
    if len(rr) < 3:
        return dict(hr=np.nan, rmssd=np.nan, sdnn=np.nan, pnn50=np.nan)

    hr = 60000 / np.median(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    sdnn = np.std(rr, ddof=1)
    pnn50 = np.mean(np.abs(np.diff(rr)) > 50) * 100

    return dict(hr=hr, rmssd=rmssd, sdnn=sdnn, pnn50=pnn50)


# ------------------------------------------------------------
# Main ECG feature extractor
# ------------------------------------------------------------
def compute_ecg_features():
    log("Computing ECG features per epoch...")

    ep_df = pd.read_parquet(EP_PARQUET)
    rows = []

    for _, r in ep_df.iterrows():
        subject = r["subject"]
        condition = r["condition"]

        ecg_path = RAW_ECG_DIR / f"{subject}_{condition}.npy"
        if not ecg_path.exists():
            log(f"Missing ECG file: {ecg_path}")
            continue

        ecg_raw = np.load(ecg_path).astype(float)

        # Epoch boundaries
        i0 = int(r["start_sec"] * ECG_FS)
        i1 = int(r["end_sec"] * ECG_FS)

        if i1 > len(ecg_raw):
            continue

        seg = ecg_raw[i0:i1]
        if len(seg) < ECG_FS:
            continue

        peaks = detect_rpeaks(seg, ECG_FS)
        rr = clean_rr(peaks, ECG_FS)
        hrv = compute_hrv(rr)

        eye_state = r.get("eye_state", None)
        state = r.get("state", eye_state)

                # ECG quality flag (non-breaking)
        ecg_ok = (
            len(peaks) >= 5 and
            len(rr) >= 3 and
            not np.isnan(hrv.get("hr", np.nan))
        )

        rows.append({
            "subject": subject,
            "condition": condition,
            "eye_state": eye_state,
            "state": state,              # alias for dashboard/RAG
            "start_sec": r["start_sec"],
            "end_sec": r["end_sec"],
            "ecg_ok": ecg_ok,            # <── NEW COLUMN (safe)
            **hrv
        })


    df = pd.DataFrame(rows)
    df.to_parquet(ECG_FEATURES_PARQ, index=False)

    log(f"Saved ECG features → {ECG_FEATURES_PARQ}")
    return df


if __name__ == "__main__":
    df = compute_ecg_features()
    print(df)
