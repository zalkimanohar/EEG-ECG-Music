"""
event_converter.py
1. Converts real event .txt files in /data into pipeline-ready CSVs
2. Normalizes filenames (lowercase, underscores)
3. Updates rec.parquet to point to correct event CSVs
4. Fixes .txt → .csv extension mismatch in rec.parquet
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# REAL event files live here (from your tree)
# ---------------------------------------------------------
REAL_EVENTS_DIR = Path("/Users/manoharazalki/EEG-ECG-Music/data")

# ---------------------------------------------------------
# Pipeline expects CSVs here
# ---------------------------------------------------------
PIPELINE_EVENTS_DIR = Path("/Users/manoharazalki/EEG-ECG-Music/LLM_RAG/artifacts/events")
PIPELINE_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

REC_PARQUET = Path("/Users/manoharazalki/EEG-ECG-Music/LLM_RAG/artifacts/rec.parquet")


# ---------------------------------------------------------
# Step 1 — Convert .txt → .csv
# ---------------------------------------------------------
def convert_event_file(txt_path):
    df = pd.read_csv(
        txt_path,
        comment="#",
        header=None,
        names=["marker", "time"]
    )
    df["marker"] = df["marker"].astype(str).str.strip()
    df["time"] = df["time"].astype(float)
    return df


def convert_all_events():
    txt_files = sorted(REAL_EVENTS_DIR.glob("*events*.txt"))

    if not txt_files:
        print("❌ No event files found in /data.")
        return

    print(f"Found {len(txt_files)} event files:")
    for f in txt_files:
        print(" -", f.name)

    print("\nConverting event files...\n")

    for txt in txt_files:
        df = convert_event_file(txt)
        out_name = txt.stem + ".csv"
        out_path = PIPELINE_EVENTS_DIR / out_name
        df.to_csv(out_path, index=False)
        print(f"✔ Converted → {out_path.name}")

    print("\n🎉 All event files converted successfully.\n")


# ---------------------------------------------------------
# Step 2 — Normalize filenames
# ---------------------------------------------------------
def normalize_name(name):
    name = name.lower()
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    return name


def normalize_event_filenames():
    print("Normalizing event filenames...\n")

    for csv in PIPELINE_EVENTS_DIR.glob("*.csv"):
        new_name = normalize_name(csv.name)
        new_path = csv.parent / new_name

        if csv.name != new_name:
            csv.rename(new_path)
            print(f"✔ Renamed: {csv.name} → {new_name}")

    print("\n✔ Filename normalization complete.\n")


# ---------------------------------------------------------
# Step 3 — Update rec.parquet (normalized names)
# ---------------------------------------------------------
def update_rec_parquet_normalized():
    print("Updating rec.parquet event paths (normalized names)...\n")

    rec = pd.read_parquet(REC_PARQUET)

    for idx, row in rec.iterrows():
        old = Path(row["events_path"]).name
        new = normalize_name(old)
        rec.at[idx, "events_path"] = str(PIPELINE_EVENTS_DIR / new)

    rec.to_parquet(REC_PARQUET, index=False)

    print("✔ rec.parquet updated with normalized names.\n")


# ---------------------------------------------------------
# Step 4 — Fix .txt → .csv extension mismatch
# ---------------------------------------------------------
def fix_rec_event_paths_extension():
    print("Fixing .txt → .csv extension mismatch in rec.parquet...\n")

    rec = pd.read_parquet(REC_PARQUET)

    for idx, row in rec.iterrows():
        old_path = Path(row["events_path"])
        stem = old_path.stem  # remove extension
        new_path = PIPELINE_EVENTS_DIR / f"{stem}.csv"
        rec.at[idx, "events_path"] = str(new_path)

    rec.to_parquet(REC_PARQUET, index=False)

    print("✔ rec.parquet updated to use .csv event files.\n")


# ---------------------------------------------------------
# Unified main()
# ---------------------------------------------------------
def main():
    convert_all_events()
    normalize_event_filenames()
    update_rec_parquet_normalized()
    fix_rec_event_paths_extension()
    print("🎉 Event conversion + normalization + path fixing complete.")
    print("Now run build_feature_database.py again.")


if __name__ == "__main__":
    main()
