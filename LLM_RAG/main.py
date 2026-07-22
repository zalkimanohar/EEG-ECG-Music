"""
main.py
Full EEG+ECG+LLM-RAG pipeline orchestrator.
"""

from config import log

from scan_dataset import scan_dataset
from load_spike_wav import load_all_spike_wav
from parse_events import parse_all_events
from eeg_features import compute_eeg_features
from ecg_features import compute_ecg_features
from build_feature_database import build_feature_database
from create_embeddings import create_embeddings
from generate_neuro_report import generate_neuro_report


def main():
    log("=== PIPELINE START ===")

    scan_dataset()
    load_all_spike_wav()
    parse_all_events()
    compute_eeg_features()
    compute_ecg_features()
    build_feature_database()
    create_embeddings()

    report = generate_neuro_report()
    print("\n=== FINAL NEURO REPORT ===\n")
    print(report)

    log("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
