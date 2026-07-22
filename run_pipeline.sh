#!/bin/bash

echo "============================================================"
echo " EEG-ECG-MUSIC — Installation Script"
echo "============================================================"

# ------------------------------------------------------------
# 1. Create virtual environment
# ------------------------------------------------------------
echo "[1/8] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# ------------------------------------------------------------
# 2. Install dependencies
# ------------------------------------------------------------
echo "[2/8] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade pip wheel setuptools

# ------------------------------------------------------------
# 3. Ensure logs directory exists
# ------------------------------------------------------------
echo "[3/7] Installing system-level dependencies..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew update
    brew install cmake
    brew install libomp
    brew install pkg-config
fi
echo "[3/8] Creating logs directory..."
mkdir -p logs

# ------------------------------------------------------------
# 4. Fix DATA_DIR in config.py
# ------------------------------------------------------------
echo "[4/7] Installing Python packages..."

pip install numpy
pip install scipy
pip install pandas
pip install matplotlib
pip install seaborn

pip install soundfile
pip install librosa

xcode-select --install
pip install watchdog
pip install streamlit

pip install torchvision
pip install sentence-transformers
pip install langchain
pip install langchain-openai

pip install faiss-cpu

pip install pyarrow
pip install fastparquet

pip install tqdm
pip install rich

pip install python-dotenv

echo "[4/8] Fixing DATA_DIR path in config.py..."

# ============================================================
# EEG + ECG + LLM‑RAG FULL PIPELINE RUNNER
# Logs stored in logs/pipeline.log
# ============================================================

# Move to project root (directory containing this script)
cd "$(dirname "$0")" || exit

rm -rf logs/pipeline.log
LOG_FILE="logs/pipeline.log"

pip install sentence-transformers

# Create logs folder if missing
mkdir -p logs


# Start logging
{
echo "============================================================"
echo "   EEG + ECG + LLM‑RAG PIPELINE RUN"
echo "   Timestamp: $(date)"
echo "============================================================"
} | tee "$LOG_FILE"

# Move into LLM_RAG directory
cd LLM_RAG || exit

run_step() {
    STEP_NAME=$1
    SCRIPT=$2

    {
        echo ""
        echo "------------------------------------------------------------"
        echo "[RUN] $STEP_NAME"
        echo "------------------------------------------------------------"
    } | tee -a "../$LOG_FILE"

    python3 "$SCRIPT" 2>&1 | tee -a "../$LOG_FILE"

    echo "[DONE] $STEP_NAME" | tee -a "../$LOG_FILE"
}

# ============================================================
# Pipeline Steps
# ============================================================

run_step "0/11 - config"              "config.py"
run_step "1/11 - Scan dataset"              "scan_dataset.py"
run_step "2/11 - Load EEG+ECG WAV files"    "load_spike_wav.py"
run_step "3/11 - Parse events"              "parse_events.py"
run_step "4/11 - Signal Processsing"        "signal_processing.py"
run_step "5/11 - Compute EEG features"      "eeg_features.py"
run_step "6/11 - Compute ECG features"      "ecg_features.py"
run_step "7/11 - Build feature database"    "build_feature_database.py"
run_step "8/11 - Train feeling model"       "ml_models.py"
run_step "9/11 - Build RAG chunks"          "build_chunks.py"
run_step "10/11 - Create embeddings"         "create_embeddings.py"
run_step "11/11 - RAG Query Engine"          "rag_query_engine.py"
run_step "12/12 - Generate neuro report"     "generate_neuro_report.py"
# run_step "13/13 - main.py"     "main.py"

{
echo ""
echo "============================================================"
echo "   PIPELINE COMPLETE"
echo "   Logs saved to logs/pipeline.log"
echo "============================================================"
} | tee -a "../$LOG_FILE"

# ============================================================
# Launch Streamlit Dashboard
# ============================================================

cd ../app || exit

echo "[RUN] Launching Streamlit dashboard..." | tee -a "../logs/pipeline.log"

streamlit run streamlit_app.py 2>&1 | tee -a "../logs/pipeline.log"
