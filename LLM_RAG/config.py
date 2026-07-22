"""
Global configuration for the EEG+ECG+LLM RAG pipeline.
Aligned with tre.txt and SpikeRecorder dataset.
"""

from pathlib import Path

# -------------------------------------------------------------------
# Project root and core folders
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # LLM_RAG/
PROJECT_ROOT = ROOT.parent                      # project root

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [DATA_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Artifact files
# -------------------------------------------------------------------
REC_PARQUET       = ARTIFACTS_DIR / "rec.parquet"
RAW_PARQUET       = ARTIFACTS_DIR / "raw.parquet"
EP_PARQUET        = ARTIFACTS_DIR / "ep.parquet"
EEG_FEATURES_PARQ = ARTIFACTS_DIR / "eeg_features.parquet"
ECG_FEATURES_PARQ = ARTIFACTS_DIR / "ecg_features.parquet"
CORPUS_PARQUET    = ARTIFACTS_DIR / "corpus.parquet"

EMBEDDINGS_NPY    = ARTIFACTS_DIR / "embeddings.npy"
EMBEDDINGS_FAISS  = ARTIFACTS_DIR / "embeddings.faiss"
METADATA_JSON     = ARTIFACTS_DIR / "metadata.json"

# -------------------------------------------------------------------
# Signal processing settings (SpikeRecorder)
# -------------------------------------------------------------------
EEG_FS = 5000
ECG_FS = 5000

# Channel mapping in WAV
ECG_CH = 0
EEG_CH = 1

# EEG frequency bands
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

ALPHA_BAND = (8, 13)
ECG_BAND   = (8, 20)

# -------------------------------------------------------------------
# Conditions
# -------------------------------------------------------------------
CONDITIONS = ["nomusic", "stress", "waterfall", "meditation"]

COND_LABELS = {
    "nomusic":   "No Music",
    "stress":    "Stress BEEP",
    "waterfall": "Waterfall Relaxation",
    "meditation": "Meditation Music",
}

# -------------------------------------------------------------------
# RAG / LLM settings
# -------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_FACTORY  = "Flat"

LLM_MODEL_NAME  = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0

# -------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------
def log(msg: str):
    print(f"[EEG_ECG_LLM_RAG] {msg}")
