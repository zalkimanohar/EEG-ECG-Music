"""
09_rag_query_engine.py
RAG engine for EEG+ECG neuro-feature corpus.
Now returns structured dicts for Streamlit + LLM.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDINGS_NPY,
    EMBEDDINGS_FAISS,
    METADATA_JSON,
    EMBEDDING_MODEL_NAME,
    log
)


class NeuroRAG:
    def __init__(self):
        log("Initializing NeuroRAG...")

        # Load embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Load FAISS index + embeddings
        self.embeddings = np.load(EMBEDDINGS_NPY)
        self.index = faiss.read_index(str(EMBEDDINGS_FAISS))

        # Load metadata (texts + rows)
        with open(METADATA_JSON, "r") as f:
            meta = json.load(f)

        self.texts = meta["texts"]          # list[str]
        self.rows = meta["rows"]            # list[dict]

    # ----------------------------------------------------------------------
    # Unified retrieval API
    # ----------------------------------------------------------------------
    def retrieve(self, query, k=10):
        """Return structured dicts: [{'text':..., 'row':..., 'score':...}, ...]"""
        return self.search(query, k=k)

    # ----------------------------------------------------------------------
    # FAISS search returning structured dicts
    # ----------------------------------------------------------------------
    def search(self, query, k=10):
        q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        D, I = self.index.search(q_emb, k)

        idxs = I[0]
        dists = D[0]

        results = []
        for rank, (idx, dist) in enumerate(zip(idxs, dists)):
            if idx < 0:
                continue

            results.append({
                "rank": rank,
                "score": float(dist),
                "text": self.texts[idx],
                "row": self.rows[idx]
            })

        return results


# ----------------------------------------------------------------------
# Helper: Build context for LLM
# ----------------------------------------------------------------------
def build_context(rag: NeuroRAG, query: str, k: int = 10):
    """
    Returns:
        context_text: concatenated text chunks
        structured_rows: list of dicts (subject, condition, hr, alpha_rel, etc.)
    """
    results = rag.retrieve(query, k=k)

    context_text = "\n\n".join([r["text"] for r in results])
    structured_rows = [r["row"] for r in results]

    return context_text, structured_rows
