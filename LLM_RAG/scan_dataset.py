"""
01_scan_dataset.py
Scans data/ for WAV + events.txt and builds rec.parquet.

Each subject has 4 WAV files:
    <name>_<condition>.wav
Each WAV has:
    <name>_<condition>_events.txt
"""

import pandas as pd
from pathlib import Path
import re
from difflib import SequenceMatcher

from config import (
    DATA_DIR,
    ARTIFACTS_DIR,
    REC_PARQUET,
    CONDITIONS,
    log
)

# ------------------------------------------------------------
# Normalize condition name
# ------------------------------------------------------------
def normalize_condition(name: str):
    name = name.lower()
    if "nomusic" in name: return "nomusic"
    if "stress" in name: return "stress"
    if "waterfall" in name: return "waterfall"
    if "meditation" in name: return "meditation"
    return None

# ------------------------------------------------------------
# Parse filename → subject + condition
# ------------------------------------------------------------
def parse_filename(filename: str):
    name = filename.replace(".wav", "")
    name = name.replace("-", "_")
    name = re.sub(r'(?<!^)([A-Z])', r'_\1', name).lower()

    parts = name.split("_")
    subject = parts[0]
    condition = normalize_condition(name)

    return subject, condition, name

# ------------------------------------------------------------
# Fuzzy match events file
# ------------------------------------------------------------
def fuzzy_find_events(wav_stem: str):
    candidates = list(DATA_DIR.glob("*events.txt"))
    best_match, best_score = None, 0.0

    for ev in candidates:
        score = SequenceMatcher(None, wav_stem.lower(), ev.stem.lower()).ratio()
        if score > best_score:
            best_score, best_match = score, ev

    return best_match if best_score > 0.55 else None

# ------------------------------------------------------------
# Main scanner
# ------------------------------------------------------------
def scan_dataset():
    log("Scanning dataset...")

    rows = []

    for file in sorted(DATA_DIR.iterdir()):
        if not file.name.endswith(".wav"):
            continue

        subject, condition, wav_stem = parse_filename(file.name)

        if condition not in CONDITIONS:
            log(f"Skipping unknown condition: {file.name}")
            continue

        events_path = fuzzy_find_events(wav_stem)
        if events_path is None:
            log(f"No events file for: {file.name}")
            continue

        rows.append(dict(
            subject=subject,
            condition=condition,
            eeg_path=str(file),
            events_path=str(events_path),
            duration_sec=None  # filled later by load_spike_wav
        ))

    df = pd.DataFrame(rows)
    df.to_parquet(REC_PARQUET, index=False)

    log(f"Saved rec.parquet → {REC_PARQUET}")
    log(f"Total recordings: {len(df)}")

    return df


if __name__ == "__main__":
    df = scan_dataset()
    print(df)
