"""
10_generate_neuro_report.py
Generate a neuroscience-style report via LLM using RAG context.
"""

from openai import OpenAI
from config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    log
)

from rag_query_engine import NeuroRAG, build_context


client = OpenAI()   # requires OPENAI_API_KEY in environment


ANALYSIS_PROMPT_TEMPLATE = """
You are a neuroscience expert.

Given the following multi-modal EEG+ECG feature summaries:

{context}

Answer in a structured way:
1. Compare stress levels across conditions (No Music, Stress BEEP, Waterfall, Meditation).
2. Explain how eye-open vs eye-closed modulates alpha and other bands.
3. Interpret heart-rate and HRV changes across conditions.
4. Provide an integrated brain–heart axis explanation.
5. Suggest how these findings relate to stress reduction and relaxation.

Use clear, technical language but keep it readable.
"""


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


def generate_neuro_report(user_query: str = None):
    rag = NeuroRAG()

    if user_query is None:
        user_query = (
            "How do different auditory environments (silence, stress-inducing sound, "
            "waterfall, meditation music) modulate human stress levels as measured "
            "through EEG, ECG, and eye-state-dependent neural response?"
        )

    context, rows = build_context(rag, user_query, k=12)
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(context=context)

    answer = query_llm(prompt)
    return answer


if __name__ == "__main__":
    report = generate_neuro_report()
    print("\n=== NEURO REPORT ===\n")
    print(report)
