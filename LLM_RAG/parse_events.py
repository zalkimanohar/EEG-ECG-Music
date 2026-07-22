"""
03_parse_events.py
Parses eye-state events for each WAV recording.

Each events.txt contains:
    1 <timestamp>
    2 <timestamp>
Where:
    1 = eyes closed
    2 = eyes open

Output:
    artifacts/ep.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    REC_PARQUET,
    ARTIFACTS_DIR,
    log
)

EP_PARQUET = ARTIFACTS_DIR / "ep.parquet"


# ------------------------------------------------------------
# Parse a single events.txt file
# ------------------------------------------------------------
def parse_event_file(path: Path):
    rows = []
    lines = path.read_text().strip().splitlines()

    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue

        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            eye = int(parts[0])
            ts = float(parts[1])
            rows.append((ts, eye))

    return rows


# ------------------------------------------------------------
# Convert raw events → epochs
# ------------------------------------------------------------
def build_epochs(events, total_duration_sec):
    if len(events) == 0:
        return []

    converted = []
    for ts, eye in events:
        ts_sec = ts / 1000.0 if ts > 1000 else ts
        converted.append((ts_sec, eye))

    converted.sort(key=lambda x: x[0])

    epochs = []
    for i in range(len(converted)):
        start = converted[i][0]
        eye = converted[i][1]

        if i < len(converted) - 1:
            end = converted[i + 1][0]
        else:
            end = total_duration_sec

        if end <= start:
            continue

        eye_state = "closed" if eye == 1 else "open"
        epochs.append((start, end, eye_state))

    return epochs


# ------------------------------------------------------------
# Main parser
# ------------------------------------------------------------
def parse_all_events():
    log("Parsing events...")

    rec_df = pd.read_parquet(REC_PARQUET)
    rows = []

    for _, r in rec_df.iterrows():
        events_path = Path(r["events_path"])

        raw_events = parse_event_file(events_path)

        duration = float(r.get("duration_sec", np.nan))
        if np.isnan(duration):
            duration = 9999

        epochs = build_epochs(raw_events, duration)

        for (start, end, eye_state) in epochs:
            rows.append(dict(
                subject=r["subject"],
                condition=r["condition"],
                start_sec=start,
                end_sec=end,
                eye_state=eye_state,
                state=eye_state,          # <── added alias for dashboard/RAG
                events_path=str(events_path)
            ))

    ep_df = pd.DataFrame(rows)
    ep_df.to_parquet(EP_PARQUET, index=False)

    log(f"Saved ep.parquet → {EP_PARQUET}")
    return ep_df


if __name__ == "__main__":
    df = parse_all_events()
    print(df)
