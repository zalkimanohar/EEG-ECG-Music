"""
10_generate_neuro_report.py
Generate a neuroscience-style report via LLM using RAG context + optional ML insights.
"""

import os
import joblib
import numpy as np
from openai import OpenAI

from config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    log
)

from rag_query_engine import NeuroRAG, build_context


# ------------------------------------------------------------
# LLM client
# ------------------------------------------------------------
client = OpenAI()   # requires OPENAI_API_KEY in environment


# ------------------------------------------------------------
# Optional ML model loading (safe fallback)
# ------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
ART = os.path.join(ROOT, "artifacts")

GLOBAL_MODEL_PATH = os.path.join(ART, "ml_global_feeling_model.pkl")
SUBJECT_MODELS_PATH = os.path.join(ART, "ml_subject_models.pkl")

def load_ml_models():
    """Load ML models if available. If not, return None safely."""
    try:
        global_data = joblib.load(GLOBAL_MODEL_PATH)
        global_model = global_data["model"]
        features = global_data["features"]

        subject_models = joblib.load(SUBJECT_MODELS_PATH)

        return global_model, subject_models, features

    except Exception:
        # ML is optional — do not break the neuro report
        return None, None, None


def ml_predict(row, global_model, subject_models, features):
    """Predict feeling state using ML (global or per-subject model)."""
    subj = row.get("subject")
    x = np.array([[row.get(f) for f in features]])

    # choose model
    if subject_models and subj in subject_models:
        model = subject_models[subj]
    else:
        model = global_model

    proba = model.predict_proba(x)[0]
    classes = model.classes_
    idx = np.argmax(proba)

    return classes[idx], float(proba[idx])


# ------------------------------------------------------------
# LLM prompt template
# ------------------------------------------------------------
ANALYSIS_PROMPT_TEMPLATE = """
You are a neuroscience expert.

Given the following multi-modal EEG+ECG feature summaries:

{context}

{ml_block}

Answer in a structured way:
1. Compare stress levels across conditions (No Music, Stress BEEP, Waterfall, Meditation).
2. Explain how eye-open vs eye-closed modulates alpha and other bands.
3. Interpret heart-rate and HRV changes across conditions.
4. Provide an integrated brain–heart axis explanation.
5. Suggest how these findings relate to stress reduction and relaxation.

Use clear, technical language but keep it readable.
"""


# ------------------------------------------------------------
# LLM query
# ------------------------------------------------------------
def query_llm(prompt: str) -> str:
    log("Querying LLM for neuro report...")

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": "You are a neuroscience expert."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


# ------------------------------------------------------------
# Main report generator
# ------------------------------------------------------------
def generate_neuro_report(user_query: str = None):
    rag = NeuroRAG()

    if user_query is None:
        user_query = (
            "How do different auditory environments (silence, stress-inducing sound, "
            "waterfall, meditation music) modulate human stress levels as measured "
            "through EEG, ECG, and eye-state-dependent neural response?"
        )

    # RAG context
    context, rows = build_context(rag, user_query, k=12)

    # ML models (optional)
    global_model, subject_models, features = load_ml_models()

    # Build ML block
    if global_model is None:
        ml_block = "\n(No ML predictions available — using RAG-only analysis.)\n"
    else:
        ml_lines = ["ML-derived feeling-state predictions:"]
        for r in rows:
            feeling, conf = ml_predict(r, global_model, subject_models, features)
            ml_lines.append(
                f"- {r['subject']} · {r['condition']}: {feeling} (confidence={conf:.2f})"
            )
        ml_block = "\n".join(ml_lines)

    # Build final prompt
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        context=context,
        ml_block=ml_block
    )

    # Query LLM
    answer = query_llm(prompt)
    return answer


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    report = generate_neuro_report()
    print("\n=== NEURO REPORT ===\n")
    print(report)
