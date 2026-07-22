"""
08_create_embeddings.py
Create text descriptions from neuro-feature corpus + RAG chunks and embed them into FAISS.

Input:
    artifacts/corpus.parquet
    artifacts/chunks.parquet

Output:
    artifacts/embeddings.npy
    artifacts/embeddings.faiss
    artifacts/metadata.json
"""

import json
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    ARTIFACTS_DIR,
    CORPUS_PARQUET,
    EMBEDDINGS_NPY,
    EMBEDDINGS_FAISS,
    METADATA_JSON,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_FACTORY,
    COND_LABELS,
    log
)


def row_to_text(row):
    cond = COND_LABELS.get(row["condition"], row["condition"])
    eye = "Closed" if row["eye_state"] == "closed" else "Open"

    return (
        f"Subject: {row['subject']}\n"
        f"Condition: {cond}\n"
        f"Eye state: {eye}\n"
        f"Epoch: {row['start_sec']:.1f}–{row['end_sec']:.1f} sec\n"
        f"EEG relative bands: "
        f"delta={row['delta_rel']:.3f}, "
        f"theta={row['theta_rel']:.3f}, "
        f"alpha={row['alpha_rel']:.3f}, "
        f"beta={row['beta_rel']:.3f}, "
        f"gamma={row['gamma_rel']:.3f}\n"
        f"ECG: HR={row['hr']:.1f} bpm, RMSSD={row['rmssd']:.1f} ms, "
        f"SDNN={row['sdnn']:.1f} ms, pNN50={row['pnn50']:.1f}%"
    )


def create_embeddings():
    log("Creating embeddings...")

    # Load original corpus
    corpus = pd.read_parquet(CORPUS_PARQUET)
    corpus_texts = [row_to_text(r) for _, r in corpus.iterrows()]

    # Load RAG chunks (if available)
    try:
        chunks = pd.read_parquet(f"{ARTIFACTS_DIR}/chunks.parquet")
        chunk_texts = chunks["chunk"].tolist()
        log(f"Loaded {len(chunk_texts)} RAG chunks.")
    except Exception:
        chunk_texts = []
        log("No RAG chunks found. Proceeding with corpus only.")

    # Combine both
    texts = corpus_texts + chunk_texts

    # Embed
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    emb = model.encode(texts, convert_to_numpy=True)

    np.save(EMBEDDINGS_NPY, emb)

    dim = emb.shape[1]
    index = faiss.index_factory(dim, FAISS_INDEX_FACTORY)
    index.add(emb.astype(np.float32))

    faiss.write_index(index, str(EMBEDDINGS_FAISS))

    # Metadata
    meta = {
        "texts": texts,
        "rows": corpus.to_dict(orient="records") + chunks.to_dict(orient="records")
    }

    with open(METADATA_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    log("Embeddings + metadata saved.")
    return emb, index, meta


if __name__ == "__main__":
    create_embeddings()
