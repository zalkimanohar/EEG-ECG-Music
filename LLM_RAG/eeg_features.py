"""
05_eeg_features.py
Extract EEG features per epoch.

Input:
    artifacts/raw_eeg/<subject>_<condition>.npy
    artifacts/ep.parquet

Output:
    artifacts/eeg_features.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    ARTIFACTS_DIR,
    EP_PARQUET,
    EEG_FS,
    EEG_CH,
    log
)

from signal_processing import (
    preprocess_eeg,
    bandpowers
)

EEG_FEATURES_PARQ = ARTIFACTS_DIR / "eeg_features.parquet"
RAW_EEG_DIR = ARTIFACTS_DIR / "raw_eeg"


def compute_eeg_features():
    log("Computing EEG features per epoch...")

    ep_df = pd.read_parquet(EP_PARQUET)
    rows = []

    for _, r in ep_df.iterrows():
        subject = r["subject"]
        condition = r["condition"]

        eeg_path = RAW_EEG_DIR / f"{subject}_{condition}.npy"
        if not eeg_path.exists():
            log(f"Missing EEG file: {eeg_path}")
            continue

        eeg_raw = np.load(eeg_path)
        eeg = preprocess_eeg(eeg_raw.astype(float), EEG_FS)

        # Epoch boundaries
        i0 = int(r["start_sec"] * EEG_FS)
        i1 = int(r["end_sec"] * EEG_FS)

        if i1 > len(eeg):
            continue

        seg = eeg[i0:i1]
        if len(seg) < EEG_FS:
            continue

        bp = bandpowers(seg, EEG_FS)

        eye_state = r.get("eye_state", None)
        state = r.get("state", eye_state)

        rows.append({
            "subject": subject,
            "condition": condition,
            "eye_state": eye_state,
            "state": state,              # <── added alias, keeps old column too
            "start_sec": r["start_sec"],
            "end_sec": r["end_sec"],
            **bp
        })

    df = pd.DataFrame(rows)
    df.to_parquet(EEG_FEATURES_PARQ, index=False)

    log(f"Saved EEG features → {EEG_FEATURES_PARQ}")
    return df


if __name__ == "__main__":
    df = compute_eeg_features()
    print(df)
