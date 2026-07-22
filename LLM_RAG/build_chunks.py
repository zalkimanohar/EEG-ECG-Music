import pandas as pd
import numpy as np

# ============================================================
# ABSOLUTE PATHS — FIX FOR FILE NOT FOUND
# ============================================================

EEG_PATH = "/Users/manoharazalki/EEG-ECG-Music/artifacts/eeg_features.parquet"
ECG_PATH = "/Users/manoharazalki/EEG-ECG-Music/artifacts/ecg_features.parquet"
REC_PATH = "/Users/manoharazalki/EEG-ECG-Music/artifacts/rec.parquet"
OUT_PATH = "/Users/manoharazalki/EEG-ECG-Music/artifacts/chunks.parquet"

# ============================================================
# BUILD RAG TEXT CHUNKS
# ============================================================

def build_chunks_main():
    print("[EEG_ECG_LLM_RAG] Building RAG text chunks...")

    # ----------------------------------------------------------
    # Load EEG + ECG + REC metadata
    # ----------------------------------------------------------
    eeg = pd.read_parquet(EEG_PATH)
    ecg = pd.read_parquet(ECG_PATH)
    rec = pd.read_parquet(REC_PATH)

    # ----------------------------------------------------------
    # Merge all three tables
    # ----------------------------------------------------------
    df = rec.merge(eeg, on=["subject", "condition"], how="left")
    df = df.merge(ecg, on=["subject", "condition", "eye_state"], how="left")

    # ----------------------------------------------------------
    # Build text chunk per row
    # ----------------------------------------------------------
    def make_chunk(row):
        subject = row.get("subject", "unknown")
        condition = row.get("condition", "unknown")
        eye = row.get("eye_state", "unknown")

        # EEG features
        alpha_rel = row.get("alpha_rel", np.nan)
        beta_rel = row.get("beta_rel", np.nan)
        theta_rel = row.get("theta_rel", np.nan)
        delta_rel = row.get("delta_rel", np.nan)
        gamma_rel = row.get("gamma_rel", np.nan)
        alpha_peak = row.get("alpha_peak", np.nan)

        # ECG features
        hr = row.get("hr", np.nan)
        rmssd = row.get("rmssd", np.nan)
        sdnn = row.get("sdnn", np.nan)
        pnn50 = row.get("pnn50", np.nan)

        # Build text chunk
        text = (
            f"subject {subject} condition {condition} eye_state {eye} "
            f"heart rate {hr} rmssd {rmssd} sdnn {sdnn} pnn50 {pnn50} "
            f"alpha_rel {alpha_rel} beta_rel {beta_rel} theta_rel {theta_rel} "
            f"delta_rel {delta_rel} gamma_rel {gamma_rel} alpha_peak {alpha_peak}"
        )

        return text

    df["chunk"] = df.apply(make_chunk, axis=1)

    # ----------------------------------------------------------
    # Save chunks
    # ----------------------------------------------------------
    chunks = df[["subject", "condition", "eye_state", "chunk"]]
    chunks.to_parquet(OUT_PATH)

    print("[EEG_ECG_LLM_RAG] ✔ RAG chunks saved →", OUT_PATH)
    print("[EEG_ECG_LLM_RAG] Total chunks:", len(chunks))


# Allow running standalone
if __name__ == "__main__":
    build_chunks_main()
