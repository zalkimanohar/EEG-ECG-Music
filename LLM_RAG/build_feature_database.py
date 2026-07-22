"""
07_build_feature_database.py
Merge EEG + ECG features into a unified neuro-feature corpus.

Input:
    artifacts/eeg_features.parquet
    artifacts/ecg_features.parquet

Output:
    artifacts/corpus.parquet
"""

import pandas as pd

from config import (
    ARTIFACTS_DIR,
    EEG_FEATURES_PARQ,
    ECG_FEATURES_PARQ,
    CORPUS_PARQUET,
    log
)


def build_feature_database():
    log("Building unified neuro-feature corpus...")

    eeg_df = pd.read_parquet(EEG_FEATURES_PARQ)
    ecg_df = pd.read_parquet(ECG_FEATURES_PARQ)

    keys = ["subject", "condition", "eye_state", "start_sec", "end_sec"]

    corpus = pd.merge(eeg_df, ecg_df, on=keys, how="inner")

    corpus.to_parquet(CORPUS_PARQUET, index=False)

    log(f"Saved corpus → {CORPUS_PARQUET}")
    log(f"Total epochs: {len(corpus)}")

    return corpus


if __name__ == "__main__":
    df = build_feature_database()
    print(df.head())
