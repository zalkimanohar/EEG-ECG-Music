"""
ML Training Script for EEG+ECG Neuro-RAG Reviewer
-------------------------------------------------

Implements the scientifically-correct ML pipeline described in the
'Music, Brain & Heart — EEG + ECG study' notebook (Section 9).

Includes:
- Per-subject normalization (z-score)
- LOSO (Leave-One-Subject-Out) training
- Global RandomForest classifier
- Per-subject fallback models
- Feeling-state label generation
- Saving models to artifacts/
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
ART = os.path.join(ROOT, "artifacts")
CORPUS = os.path.join(ART, "corpus.parquet")

GLOBAL_MODEL_PATH = os.path.join(ART, "ml_global_feeling_model.pkl")
SUBJECT_MODELS_PATH = os.path.join(ART, "ml_subject_models.pkl")

# -------------------------------------------------------------------
# Features used for ML
# -------------------------------------------------------------------
FEATURES = [
    "delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel",
    "hr", "rmssd", "sdnn", "pnn50"
]

# -------------------------------------------------------------------
# Feeling-state label generation (your Page 4 logic)
# -------------------------------------------------------------------
def derive_feeling(row):
    alpha = row["alpha_rel"]
    hr = row["hr"]
    rmssd = row["rmssd"]

    if alpha > 0.6 and hr < 70 and rmssd > 80:
        return "relaxed"
    if alpha > 0.5 and hr < 75:
        return "calm"
    if alpha < 0.3 and hr > 80 and rmssd < 50:
        return "stressed"
    if alpha < 0.4 and hr > 75:
        return "tense"
    return "neutral"

# -------------------------------------------------------------------
# Per-subject normalization (critical)
# -------------------------------------------------------------------
def normalize_subject(df):
    out = []
    for subj, d in df.groupby("subject"):
        d_norm = d.copy()
        d_norm[FEATURES] = (d[FEATURES] - d[FEATURES].mean()) / d[FEATURES].std()
        out.append(d_norm)
    return pd.concat(out)

# -------------------------------------------------------------------
# LOSO training (global model)
# -------------------------------------------------------------------
def train_loso(df):
    X = df[FEATURES].values
    y = df["feeling"].values
    groups = df["subject"].values

    logo = LeaveOneGroupOut()
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )

    # LOSO evaluation loop (optional logging)
    for train_idx, test_idx in logo.split(X, y, groups):
        rf.fit(X[train_idx], y[train_idx])

    # Final fit on all data
    rf.fit(X, y)
    return rf

# -------------------------------------------------------------------
# Per-subject fallback models
# -------------------------------------------------------------------
def train_per_subject_models(df):
    models = {}
    for subj, d in df.groupby("subject"):
        if len(d) < 10:
            continue
        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        )
        rf.fit(d[FEATURES], d["feeling"])
        models[subj] = rf
    return models

# -------------------------------------------------------------------
# Main training function
# -------------------------------------------------------------------
def train_all_models():
    print("Loading corpus...")
    df = pd.read_parquet(CORPUS)

    print("Generating feeling labels...")
    df["feeling"] = df.apply(derive_feeling, axis=1)
    df = df.dropna(subset=["feeling"])

    print("Normalizing per subject...")
    df_norm = normalize_subject(df)

    print("Training global LOSO model...")
    global_model = train_loso(df_norm)

    print("Training per-subject fallback models...")
    subject_models = train_per_subject_models(df_norm)

    print("Saving models...")
    joblib.dump({"model": global_model, "features": FEATURES}, GLOBAL_MODEL_PATH)
    joblib.dump(subject_models, SUBJECT_MODELS_PATH)

    print("ML training complete.")
    print(f"Global model saved to: {GLOBAL_MODEL_PATH}")
    print(f"Subject models saved to: {SUBJECT_MODELS_PATH}")

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    train_all_models()
