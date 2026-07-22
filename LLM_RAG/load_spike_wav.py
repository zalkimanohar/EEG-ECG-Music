"""
02_load_spike_wav.py
Loads EEG+ECG WAV files from rec.parquet and stores raw signals.

Output:
- artifacts/raw_eeg/<subject>_<condition>.npy
- artifacts/raw_ecg/<subject>_<condition>.npy
- artifacts/raw.parquet
"""

import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path

from config import (
    ARTIFACTS_DIR,
    REC_PARQUET,
    EEG_CH,
    ECG_CH,
    log
)

RAW_EEG_DIR = ARTIFACTS_DIR / "raw_eeg"
RAW_ECG_DIR = ARTIFACTS_DIR / "raw_ecg"

RAW_EEG_DIR.mkdir(parents=True, exist_ok=True)
RAW_ECG_DIR.mkdir(parents=True, exist_ok=True)


def load_all_spike_wav():
    log("Loading EEG+ECG WAV files...")

    df = pd.read_parquet(REC_PARQUET)
    rows = []

    for _, r in df.iterrows():
        wav_path = Path(r["eeg_path"])

        if not wav_path.exists():
            log(f"Missing WAV: {wav_path}")
            continue

        try:
            signal, fs = sf.read(wav_path)

            if signal.ndim == 1:
                signal = np.column_stack([signal, signal])

            ecg = signal[:, ECG_CH]
            eeg = signal[:, EEG_CH]

            base = f"{r['subject']}_{r['condition']}"
            eeg_out = RAW_EEG_DIR / f"{base}.npy"
            ecg_out = RAW_ECG_DIR / f"{base}.npy"

            np.save(eeg_out, eeg)
            np.save(ecg_out, ecg)

            rows.append(dict(
                subject=r["subject"],
                condition=r["condition"],
                eeg_raw_path=str(eeg_out),
                ecg_raw_path=str(ecg_out),
                fs=fs,
                duration_sec=len(eeg) / fs
            ))

        except Exception as e:
            log(f"Failed to load {wav_path}: {e}")

    raw_df = pd.DataFrame(rows)
    raw_df.to_parquet(ARTIFACTS_DIR / "raw.parquet", index=False)

    log("Saved raw.parquet")
    return raw_df


if __name__ == "__main__":
    df = load_all_spike_wav()
    print(df)
