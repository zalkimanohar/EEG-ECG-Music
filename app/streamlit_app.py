import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import spectrogram

# ----------------------------------------------------------
# Paths aligned with your actual project tree
# ----------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent          # .../EEG-ECG-Music/app
ROOT = APP_DIR.parent                              # .../EEG-ECG-Music
ART = ROOT / "artifacts"                           # .../EEG-ECG-Music/artifacts
LLM = ROOT / "LLM_RAG"                             # .../EEG-ECG-Music/LLM_RAG

sys.path.append(str(ROOT))
sys.path.append(str(LLM))

# ----------------------------------------------------------
# Imports from LLM_RAG
# ----------------------------------------------------------
from rag_query_engine import NeuroRAG
from signal_processing import (
    preprocess_eeg,
    alpha_envelope,
    detect_rpeaks,
    clean_rr,
    find_rec,
    bandpass_ecg,
)

# ----------------------------------------------------------
# Colors & constants
# ----------------------------------------------------------
INK   = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID  = "#e1e0d9"; AXIS  = "#c3c2b7"

C_COND = {
    "nomusic":    "#2a78d6",
    "meditation": "#008300",
    "waterfall":  "#eda100",
    "stress":     "#e34948",
}
C_STATE = {"closed": "#1c5cab", "open": "#eb6834"}
C_ALPHA = "#eb6834"

CONDITIONS = ["nomusic", "meditation", "waterfall", "stress"]
COND_EN = {
    "nomusic": "No Music",
    "meditation": "Meditation",
    "waterfall": "Waterfall",
    "stress": "Stress",
}

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def style(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.9)
    ax.grid(False, axis="x" if grid_axis == "y" else "y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(length=0)
    return ax

def compute_hrv_metrics(rr_ms: np.ndarray):
    rr = np.asarray(rr_ms, dtype=float)
    if len(rr) < 3:
        return dict(rmssd=np.nan, sdnn=np.nan, pnn50=np.nan)
    diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff**2))
    sdnn = np.std(rr, ddof=1)
    pnn50 = np.sum(np.abs(diff) > 50.0) / len(diff) * 100.0
    return dict(rmssd=rmssd, sdnn=sdnn, pnn50=pnn50)

def get_alpha_col(df):
    for c in ["Alpha_rel", "alpha_rel", "alpha"]:
        if c in df.columns:
            return c
    return None

# ----------------------------------------------------------
# Load parquet tables (aligned with artifacts/)
# ----------------------------------------------------------
@st.cache_data
def load_tables():
    rec = pd.read_parquet(ART / "rec.parquet")
    eeg_feat = pd.read_parquet(ART / "eeg_features.parquet")
    ep = pd.read_parquet(ART / "ep.parquet")
    corpus = pd.read_parquet(ART / "corpus.parquet")
    return rec, eeg_feat, ep, corpus

rec, eeg_feat, ep, corpus = load_tables()

# ----------------------------------------------------------
# Alpha source selection (robust, no pipeline impact)
# ----------------------------------------------------------
def load_alpha_source():
    if not eeg_feat.empty:
        alpha_col = get_alpha_col(eeg_feat)
        if alpha_col:
            df = eeg_feat.copy()
            if "state" not in df.columns:
                df["state"] = "closed"
            return df, alpha_col
    if not ep.empty:
        alpha_col = get_alpha_col(ep)
        if alpha_col:
            df = ep.copy()
            if "state" not in df.columns:
                df["state"] = "closed"
            return df, alpha_col
    return None, None

alpha_source, alpha_col = load_alpha_source()

# ----------------------------------------------------------
# Load RAG engine
# ----------------------------------------------------------
@st.cache_resource
def load_rag():
    return NeuroRAG()

engine = load_rag()

# ----------------------------------------------------------
# Streamlit config
# ----------------------------------------------------------
st.set_page_config(
    page_title="Music, Brain & Heart — Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🎧 Music, Brain & Heart Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Global Insights", "Subject Explorer", "Multi-Subject Comparison", "LLM‑RAG Neuro Explorer"],
)

# ============================================================
# PAGE 1 — GLOBAL INSIGHTS (FINAL WORKING VERSION)
# ============================================================
if page == "Global Insights":
    st.title("📊 Global Insights — Complete Dataset")

    # ----------------------------------------------------------
    # Summary metrics
    # ----------------------------------------------------------
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Participants", rec["subject"].nunique())
    colB.metric("Recordings", len(rec))
    colC.metric("Conditions", rec["condition"].nunique())
    colD.metric("Good ECG", rec["ecg_ok"].sum() if "ecg_ok" in rec.columns else "N/A")

    st.markdown("### 📐 Statistical Tests on Alpha Power")

    import scipy.stats as stats

    # ----------------------------------------------------------
    # Alpha statistical tests
    # ----------------------------------------------------------
    if alpha_source is None or alpha_col is None:
        st.info("No alpha column found — showing ECG insights only.")
    else:
        ep_state = alpha_source.copy()
        if "state" not in ep_state.columns:
            ep_state["state"] = "closed"

        ep_state = ep_state[ep_state["state"].isin(["closed", "open"])]
        ep_closed = ep_state[ep_state["state"] == "closed"]

        # Wilcoxon test
        piv = ep_state.pivot_table(index="subject", columns="state",
                                   values=alpha_col, aggfunc="mean")

        wil_p = None
        if {"closed", "open"}.issubset(piv.columns):
            try:
                wil_p = stats.wilcoxon(piv["closed"], piv["open"]).pvalue
            except Exception:
                wil_p = None

        # Friedman test
        pv = ep_closed.pivot_table(index="subject", columns="condition",
                                   values=alpha_col, aggfunc="mean")
        pv = pv.reindex(columns=["nomusic", "stress", "waterfall", "meditation"])
        pv_clean = pv.dropna(how="any")

        fr_p = None
        if len(pv_clean) >= 3:
            try:
                fr_p = stats.friedmanchisquare(
                    pv_clean["nomusic"],
                    pv_clean["stress"],
                    pv_clean["waterfall"],
                    pv_clean["meditation"],
                ).pvalue
            except Exception:
                fr_p = None

        colE, colF = st.columns(2)
        colE.metric("Wilcoxon p (Closed vs Open)", f"{wil_p:.4f}" if wil_p else "N/A")
        colF.metric("Friedman p (Across Conditions)", f"{fr_p:.4f}" if fr_p else "N/A")

        # ------------------------------------------------------
        # Berger Effect Plot
        # ------------------------------------------------------
        st.subheader("🧠 Berger Effect — Alpha (Eyes Closed vs Open)")
        alpha_state = ep_state.groupby("state")[alpha_col].mean()

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(alpha_state.index, alpha_state.values,
               color=[C_STATE[s] for s in alpha_state.index])
        ax.set_ylabel("Mean Alpha (rel)")
        ax.set_title("Alpha Increase When Eyes Close", loc="left")
        style(ax)
        st.pyplot(fig)

        # ------------------------------------------------------
        # Alpha by Condition
        # ------------------------------------------------------
        st.subheader("🎼 Alpha (Eyes Closed) by Music Condition")
        alpha_cond = ep_closed.groupby("condition")[alpha_col].mean()

        fig, ax = plt.subplots(figsize=(7, 3))
        colors = [C_COND[c] for c in alpha_cond.index]
        ax.bar([COND_EN[c] for c in alpha_cond.index],
               alpha_cond.values, color=colors)
        ax.set_ylabel("Mean Alpha (rel)")
        ax.set_title("Relaxation Signature Across Sound Conditions", loc="left")
        style(ax)
        st.pyplot(fig)

    # ----------------------------------------------------------
    # ⭐ SUBJECT-WISE ALPHA SCATTER
    # ----------------------------------------------------------
    st.subheader("🧑‍🏫 Subject-wise Alpha (Eyes Closed) — Each Dot is One Person")

    if alpha_source is not None and alpha_col is not None:
        ep_closed = alpha_source[alpha_source["state"] == "closed"]
        metric = ep_closed.groupby("subject")[alpha_col].mean()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(metric.index, metric.values,
                   s=140, color=C_ALPHA, edgecolor="white", linewidth=1.4)

        for s, v in metric.items():
            ax.text(s, v + 0.005, f"{v:.2f}", ha="center", fontsize=9, color=INK2)

        ax.set_ylabel("Relative Alpha (Eyes Closed)")
        ax.set_title("Inter-Subject Variability in Relaxation Response", loc="left")
        ax.set_xticks(range(len(metric.index)))
        ax.set_xticklabels(metric.index, rotation=45)
        style(ax)
        st.pyplot(fig)

    # ----------------------------------------------------------
    # ⭐ CONDITION-WISE ALPHA (BAR + SCATTER + SD)
    # ----------------------------------------------------------
    st.subheader("📊 Condition-wise Alpha — Mean, Variability & Individuals")

    if alpha_source is not None and alpha_col is not None:
        ep_closed = alpha_source[alpha_source["state"] == "closed"]

        pv = ep_closed.pivot_table(
            index="subject",
            columns="condition",
            values=alpha_col,
            aggfunc="mean",
        ).reindex(columns=["nomusic", "stress", "waterfall", "meditation"])

        means = pv.mean()
        stds  = pv.std()

        fig, ax = plt.subplots(figsize=(10, 4))
        xpos = np.arange(len(means))

        ax.bar(
            xpos, means.values, yerr=stds.values, capsize=6,
            color=[C_COND[c] for c in means.index],
            alpha=0.75, edgecolor="black"
        )

        for i, c in enumerate(means.index):
            ax.scatter(np.repeat(i, len(pv)), pv[c].values,
                       color="black", s=60, alpha=0.85,
                       edgecolor="white", linewidth=1.2)

        ax.set_xticks(xpos)
        ax.set_xticklabels([COND_EN[c] for c in means.index])
        ax.set_ylabel("Relative Alpha (Eyes Closed)")
        ax.set_title("Alpha Power Across Conditions — Mean, Variability & Individuals", loc="left")
        style(ax)
        st.pyplot(fig)

    # ----------------------------------------------------------
    # ⭐ ECG GRAPH 1 — SUBJECT-WISE HEART RATE
    # ----------------------------------------------------------
    st.subheader("❤️ Subject-wise Heart Rate — Each Dot is One Person")

    ecg_feat = pd.read_parquet(ART / "ecg_features.parquet")
    hr_mean = ecg_feat.groupby("subject")["hr"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(hr_mean.index, hr_mean.values,
               s=140, color="#e34948", edgecolor="white", linewidth=1.4)

    for s, v in hr_mean.items():
        ax.text(s, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, color=INK2)

    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title("Inter-Subject Variability in Heart Rate", loc="left")
    ax.set_xticks(range(len(hr_mean.index)))
    ax.set_xticklabels(hr_mean.index, rotation=45)
    style(ax)
    st.pyplot(fig)

    # ----------------------------------------------------------
    # ⭐ ECG GRAPH 2 — CONDITION-WISE HEART RATE (BAR + SCATTER + SD)
    # ----------------------------------------------------------
    st.subheader("💓 Condition-wise Heart Rate — Mean, Variability & Individuals")

    pv_hr = ecg_feat.pivot_table(
        index="subject",
        columns="condition",
        values="hr",
        aggfunc="mean"
    ).reindex(columns=["nomusic", "stress", "waterfall", "meditation"])

    means = pv_hr.mean()
    stds  = pv_hr.std()

    fig, ax = plt.subplots(figsize=(10, 4))
    xpos = np.arange(len(means))

    ax.bar(
        xpos, means.values, yerr=stds.values, capsize=6,
        color=[C_COND[c] for c in means.index],
        alpha=0.75, edgecolor="black"
    )

    for i, c in enumerate(means.index):
        ax.scatter(np.repeat(i, len(pv_hr)), pv_hr[c].values,
                   color="black", s=60, alpha=0.85,
                   edgecolor="white", linewidth=1.2)

    ax.set_xticks(xpos)
    ax.set_xticklabels([COND_EN[c] for c in means.index])
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title("Heart Rate Across Conditions — Mean, Variability & Individuals", loc="left")
    style(ax)
    st.pyplot(fig)

# ============================================================
# PAGE 2 — SUBJECT EXPLORER
# ============================================================
elif page == "Subject Explorer":
    st.title("🧑‍🔬 Subject & Condition Explorer")

    subjects = sorted(rec["subject"].unique())
    sel_subj = st.sidebar.selectbox("Subject", subjects)

    condition_options = ["all"] + CONDITIONS
    sel_cond = st.sidebar.selectbox(
        "Condition",
        condition_options,
        format_func=lambda c: "ALL" if c == "all" else COND_EN[c],
    )

    # Custom order for Page 2
    ORDER = ["nomusic", "stress", "waterfall", "meditation"]

    if sel_cond == "all":
        st.subheader(f"Recordings for {sel_subj}")
        subj_recs = rec[rec["subject"] == sel_subj].copy()
        st.dataframe(subj_recs)

        if alpha_source is not None and alpha_col is not None:
            ep_closed = alpha_source[alpha_source["state"] == "closed"]

            pv = ep_closed[ep_closed["subject"] == sel_subj].pivot_table(
                index="subject",
                columns="condition",
                values=alpha_col,
                aggfunc="mean",
            )

            # Apply custom order
            pv = pv.reindex(columns=ORDER)

            fig, ax = plt.subplots(figsize=(7, 4))
            xpos = np.arange(len(ORDER))
            vals = pv.loc[sel_subj].values if sel_subj in pv.index else np.zeros(len(ORDER))

            ax.plot(xpos, vals, "-o", color=INK2)

            for i, c in enumerate(ORDER):
                ax.scatter(i, vals[i], s=60, color=C_COND[c],
                           edgecolor="white", linewidth=1)

            ax.set_xticks(xpos)
            ax.set_xticklabels([COND_EN[c] for c in ORDER])
            ax.set_ylabel("Mean Alpha (Eyes Closed)")
            ax.set_title(f"{sel_subj} — Alpha Across Conditions")
            style(ax)
            st.pyplot(fig)

        else:
            st.info("Alpha source missing — cannot plot alpha across conditions.")

        st.info("Select a single condition to view EEG/ECG anatomy, spectrograms, and HRV.")
        st.stop()

    # ----------------------------------------------------------
    # Single-condition detailed explorer
    # ----------------------------------------------------------
    fs, eeg_raw, fs_ecg, ecg_raw, ev = find_rec(sel_subj, sel_cond)
    eeg = preprocess_eeg(eeg_raw, fs)
    env = alpha_envelope(eeg, fs)
    ecg_bp = bandpass_ecg(ecg_raw, fs_ecg) if ecg_raw is not None else None
    peaks = detect_rpeaks(ecg_raw, fs_ecg) if ecg_raw is not None else np.array([])

    t = np.arange(len(eeg)) / fs
    t_eeg_raw = np.arange(len(eeg_raw)) / fs
    t_ecg = np.arange(len(ecg_raw)) / fs_ecg if ecg_raw is not None else None

    st.subheader("🧠 EEG Anatomy (Raw, Filtered, Alpha Envelope)")
    fig, axs = plt.subplots(3, 1, figsize=(11, 6), sharex=True,
                            gridspec_kw=dict(hspace=0.18))

    axs[0].plot(t_eeg_raw, eeg_raw - eeg_raw.mean(), color=MUTED, lw=0.4)
    axs[0].set_ylabel("Raw EEG")
    style(axs[0])

    axs[1].plot(t, eeg, color=C_COND[sel_cond], lw=0.5)
    axs[1].set_ylabel("Filtered EEG")
    style(axs[1])

    axs[2].plot(t, env, color=C_ALPHA, lw=1.6)
    axs[2].fill_between(t, 0, env, color=C_ALPHA, alpha=0.15)
    axs[2].set_ylabel("Alpha Envelope")
    axs[2].set_xlabel("Time (s)")
    style(axs[2])

    st.pyplot(fig)

    st.subheader("💓 ECG Anatomy (Raw, Bandpass, R-peaks)")
    if ecg_raw is not None and ecg_bp is not None:
        fig, axs = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                                gridspec_kw=dict(hspace=0.18))

        axs[0].plot(t_ecg, ecg_raw - ecg_raw.mean(), color=MUTED, lw=0.5)
        axs[0].set_ylabel("Raw ECG")
        style(axs[0])

        axs[1].plot(t_ecg, ecg_bp, color=C_COND[sel_cond], lw=0.7)
        if len(peaks) > 0:
            axs[1].scatter(peaks / fs_ecg, ecg_bp[peaks],
                           color="#e34948", s=18, label="R-peaks")
            axs[1].legend(loc="upper right")
        axs[1].set_ylabel("Bandpass ECG")
        axs[1].set_xlabel("Time (s)")
        style(axs[1])

        st.pyplot(fig)
    else:
        st.info("ECG not available for this recording.")

    st.subheader("📡 EEG Spectrogram (Interactive Zoom)")
    f, tt, Sxx = spectrogram(eeg, fs, nperseg=1024, noverlap=896)
    S = 10 * np.log10(Sxx + 1e-12)

    max_time = float(tt.max())
    max_freq = float(f.max())

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        t_min = st.slider("Time min (s)", 0.0, max_time, 0.0, 0.5)
    with col_t2:
        t_max = st.slider("Time max (s)", 0.5, max_time, max_time, 0.5)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_min = st.slider("Freq min (Hz)", 0.0, max_freq, 0.0, 1.0)
    with col_f2:
        f_max = st.slider("Freq max (Hz)", 5.0, max_freq, min(30.0, max_freq), 1.0)

    t_mask = (tt >= t_min) & (tt <= t_max)
    f_mask = (f >= f_min) & (f <= f_max)

    fig, ax = plt.subplots(figsize=(11, 4))
    pcm = ax.pcolormesh(tt[t_mask], f[f_mask], S[f_mask][:, t_mask],
                        cmap="magma", shading="gouraud",
                        vmin=np.percentile(S[f_mask][:, t_mask], 5),
                        vmax=np.percentile(S[f_mask][:, t_mask], 99))
    ax.axhline(8, color="white", lw=0.6, ls=":", alpha=0.6)
    ax.axhline(12, color="white", lw=0.6, ls=":", alpha=0.6)
    ax.set_ylim(f_min, f_max)
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("EEG Spectrogram (Zoomed)")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(pcm, cax=cax, label="dB")
    st.pyplot(fig)

    st.subheader("💓 HRV — Poincaré Plot & Metrics")
    rr = clean_rr(peaks, fs_ecg)
    hrv = compute_hrv_metrics(rr)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("RMSSD (ms)", f"{hrv['rmssd']:.1f}" if not np.isnan(hrv['rmssd']) else "N/A")
    col_m2.metric("SDNN (ms)", f"{hrv['sdnn']:.1f}" if not np.isnan(hrv['sdnn']) else "N/A")
    col_m3.metric("pNN50 (%)", f"{hrv['pnn50']:.1f}" if not np.isnan(hrv['pnn50']) else "N/A")

    fig, ax = plt.subplots(figsize=(5, 4))
    if len(rr) > 2:
        ax.scatter(rr[:-1], rr[1:], s=26, color=C_COND[sel_cond],
                   alpha=0.7, edgecolor="white", linewidth=0.4)
    ax.plot([400, 1200], [400, 1200], color=AXIS, lw=1, ls="--")
    ax.set_xlim(400, 1200)
    ax.set_ylim(400, 1200)
    ax.set_aspect("equal")
    ax.set_xlabel("RRₙ (ms)")
    ax.set_ylabel("RRₙ₊₁ (ms)")
    ax.set_title("Poincaré Plot")
    style(ax, grid_axis="both")
    st.pyplot(fig)
# ============================================================
# PAGE 3 — MULTI-SUBJECT COMPARISON (FULLY REVIEWED & CUSTOMIZED)
# ============================================================
elif page == "Multi-Subject Comparison":
    st.title("👥 Multi-Subject Comparison Dashboards")

    # ----------------------------------------------------------
    # Fixed condition order (consistent across dashboard)
    # ----------------------------------------------------------
    CONDITION_ORDER = ["nomusic", "stress", "waterfall", "meditation"]

    COND_EN_ORDER = {
        "nomusic": "No Music",
        "stress": "Stress",
        "waterfall": "WaterFall",
        "meditation": "Meditation"
    }

    subjects = sorted(rec["subject"].unique())
    sel_subjects = st.sidebar.multiselect(
        "Select subjects",
        subjects,
        default=subjects[:4]
    )

    if not sel_subjects:
        st.info("Select at least one subject.")
        st.stop()

    # ----------------------------------------------------------
    # 🧠 Alpha (Eyes Closed) Across Subjects & Conditions
    # ----------------------------------------------------------
    st.subheader("🧠 Alpha (Eyes Closed) Across Subjects & Conditions")
    st.markdown(
        "This plot shows **how each selected subject’s alpha power changes across sound conditions**. "
        "Each line represents one subject."
    )

    if alpha_source is not None and alpha_col is not None:
        ep_closed = alpha_source[alpha_source["state"] == "closed"]
        df_alpha = ep_closed[ep_closed["subject"].isin(sel_subjects)]

        pv = df_alpha.pivot_table(
            index="subject",
            columns="condition",
            values=alpha_col,
            aggfunc="mean",
        ).reindex(columns=CONDITION_ORDER)

        fig, ax = plt.subplots(figsize=(10, 4))
        xpos = np.arange(len(CONDITION_ORDER))

        for s in sel_subjects:
            if s in pv.index:
                vals = pv.loc[s].values
                ax.plot(xpos, vals, "-o", label=str(s), linewidth=1.8)

        ax.set_xticks(xpos)
        ax.set_xticklabels([COND_EN_ORDER[c] for c in CONDITION_ORDER])
        ax.set_ylabel("Mean Alpha (Eyes Closed)")
        ax.set_title("Alpha Across Conditions — Selected Subjects", loc="left")
        ax.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left")
        style(ax)
        st.pyplot(fig)

    else:
        st.info("Alpha source missing — cannot plot multi-subject alpha.")
    
    # ----------------------------------------------------------
    # 💓 Heart Rate Across Subjects & Conditions
    # ----------------------------------------------------------
    st.subheader("💓 Heart Rate Across Subjects & Conditions")
    st.markdown(
        "This plot shows **how heart rate changes across conditions** for each selected subject."
    )

    try:
        ecg_feat = pd.read_parquet(ART / "ecg_features.parquet")
        df_hr = ecg_feat[ecg_feat["subject"].isin(sel_subjects)]

        if "hr" not in df_hr.columns:
            st.error("HR column missing in ecg_features.parquet.")
            st.stop()

        pv_hr = df_hr.pivot_table(
            index="subject",
            columns="condition",
            values="hr",
            aggfunc="mean"
        ).reindex(columns=CONDITION_ORDER)

        fig, ax = plt.subplots(figsize=(10, 4))
        xpos = np.arange(len(CONDITION_ORDER))

        for s in sel_subjects:
            if s in pv_hr.index:
                vals = pv_hr.loc[s].values
                ax.plot(xpos, vals, "-o", label=str(s), linewidth=1.8)

        ax.set_xticks(xpos)
        ax.set_xticklabels([COND_EN_ORDER[c] for c in CONDITION_ORDER])
        ax.set_ylabel("Heart Rate (bpm)")
        ax.set_title("Heart Rate Across Conditions — Selected Subjects", loc="left")
        ax.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left")
        style(ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not load HR data: {e}")

# ============================================================
# PAGE 4 — LLM‑RAG NEURO AGENT REVIEWER (FINAL POLISHED VIEW)
# ============================================================
elif page == "LLM‑RAG Neuro Explorer":
    from langchain_openai import ChatOpenAI
    import numpy as np
    import joblib
    import os
    import pandas as pd

    st.markdown(
        """
        <style>
        .chat-bubble-user {
            background-color:#1e1e1e;
            padding:12px;
            border-radius:10px;
            margin-bottom:10px;
            border:1px solid #444;
        }
        .chat-bubble-assistant {
            background-color:#262626;
            padding:12px;
            border-radius:10px;
            margin-bottom:10px;
            border:1px solid #555;
        }
        .summary-card {
            padding:15px;
            border-radius:10px;
            background-color:#333;
            margin-bottom:15px;
            border:1px solid #555;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧬 Biomedical Research Assistant — Neuro‑RAG Reviewer")

    # ----------------------------------------------------------
    # Load ML models (safe fallback)
    # ----------------------------------------------------------
    ROOT = os.path.dirname(os.path.dirname(__file__))
    ART = os.path.join(ROOT, "artifacts")

    GLOBAL_MODEL_PATH = os.path.join(ART, "ml_global_feeling_model.pkl")
    SUBJECT_MODELS_PATH = os.path.join(ART, "ml_subject_models.pkl")

    def load_ml_models():
        try:
            global_data = joblib.load(GLOBAL_MODEL_PATH)
            global_model = global_data["model"]
            ml_features = global_data["features"]
            subject_models = joblib.load(SUBJECT_MODELS_PATH)
            return global_model, subject_models, ml_features
        except Exception:
            return None, None, None

    global_model, subject_models, ml_features = load_ml_models()

    # ----------------------------------------------------------
    # ML prediction helper
    # ----------------------------------------------------------
    def ml_predict(row):
        if global_model is None or ml_features is None:
            return None, None

        subj = row.get("subject")
        x = np.array([[row.get(f) for f in ml_features]])

        if subject_models and subj in subject_models:
            model = subject_models[subj]
        else:
            model = global_model

        proba = model.predict_proba(x)[0]
        classes = model.classes_
        idx = int(np.argmax(proba))

        return classes[idx], float(proba[idx])

    FEELING_ICONS = {
        "relaxed": "🟢",
        "calm": "🟦",
        "neutral": "⚪",
        "tense": "🟧",
        "stressed": "🔴",
    }

    # ----------------------------------------------------------
    # Conversation state
    # ----------------------------------------------------------
    if "rag_chat" not in st.session_state:
        st.session_state.rag_chat = []

    subjects = sorted(rec["subject"].unique())
    sel_subject = st.sidebar.selectbox("Subject filter", ["All"] + subjects)
    sel_conditions = st.sidebar.multiselect("Conditions", CONDITIONS, default=CONDITIONS)
    k = st.sidebar.slider("Retrieved contexts", 3, 20, 10)

    # ML indicator
    if global_model is None:
        st.sidebar.markdown("⚠️ **ML Model:** Not Loaded (RAG‑only mode)")
    else:
        st.sidebar.markdown("🔬 **ML Model:** Active (LOSO + per‑subject calibration)")

    # ----------------------------------------------------------
    # Chat bubble renderer
    # ----------------------------------------------------------
    def render_chat_bubble(role, content):
        if role == "user":
            css_class = "chat-bubble-user"
            label = "You"
        else:
            css_class = "chat-bubble-assistant"
            label = "Biomedical Research Assistant"

        st.markdown(
            f"""
            <div class="{css_class}">
                <strong>{label}:</strong><br>{content}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💬 Ask the Biomedical Research Assistant")

    # Conversation history
    for msg in st.session_state.rag_chat:
        render_chat_bubble(msg["role"], msg["content"])

    user_input = st.text_area(
        "Ask anything about EEG/ECG reactions, feeling, blind spots, or your research question:",
        "",
        key="rag_user_input",
    )

    col_send, col_clear = st.columns(2)
    send_clicked = col_send.button("Send")
    clear_clicked = col_clear.button("Clear Conversation")

    if clear_clicked:
        st.session_state.rag_chat = []
        st.rerun()

    # ----------------------------------------------------------
    # Detect subject mentioned in question
    # ----------------------------------------------------------
    def detect_subject_from_question(q):
        q = q.lower()
        for s in subjects:
            if s.lower() in q:
                return s
        return None

    asked_subject = detect_subject_from_question(user_input)

    # ----------------------------------------------------------
    # MAIN EXECUTION
    # ----------------------------------------------------------
    if send_clicked and user_input.strip():
        st.session_state.rag_chat.append({"role": "user", "content": user_input})

        # 1) Retrieve structured dicts from RAG
        results = engine.retrieve(user_input, k=k)

        # 2) Filter by subject (sidebar + question)
        if sel_subject != "All":
            results = [r for r in results if r["row"].get("subject") == sel_subject]

        if asked_subject is not None:
            results = [r for r in results if r["row"].get("subject") == asked_subject]

        # 3) Filter by condition
        results = [r for r in results if r["row"].get("condition") in sel_conditions]

        # 4) Strict RAG fallback
        if len(results) == 0:
            fallback_answer = (
                "### 1) Blind Spots\n"
                "No EEG/ECG records were retrieved for the requested subject or conditions.\n\n"
                "### 2) General Physiology (Not From Your Data)\n"
                "- Meditation → alpha↑, HR↓, HRV↑ (relaxation)\n"
                "- Stress → alpha↓, HR↑, HRV↓ (sympathetic activation)\n\n"
                "### 3) Interpretation\n"
                "Your question cannot be answered strictly from available data.\n\n"
                "### 4) Conclusion\n"
                "Add more EEG/ECG chunks for this subject/condition to enable strict RAG analysis."
            )
            st.session_state.rag_chat.append({"role": "assistant", "content": fallback_answer})
            render_chat_bubble("assistant", fallback_answer)
            st.rerun()

        # ----------------------------------------------------------
        # 5) Build per-condition physiological + ML summary
        # ----------------------------------------------------------
        def safe_mean(values):
            arr = [v for v in values if v is not None]
            return float(np.nanmean(arr)) if len(arr) > 0 else np.nan

        cond_summary_structured = {}
        ml_summary_structured = {}

        for r in results:
            row = r["row"]
            cond = row.get("condition")

            # Physiological summary
            if cond not in cond_summary_structured:
                cond_summary_structured[cond] = {
                    "hr": [], "alpha_rel": [], "beta_rel": [],
                    "rmssd": [], "sdnn": []
                }

            cond_summary_structured[cond]["hr"].append(row.get("hr"))
            cond_summary_structured[cond]["alpha_rel"].append(row.get("alpha_rel"))
            cond_summary_structured[cond]["beta_rel"].append(row.get("beta_rel"))
            cond_summary_structured[cond]["rmssd"].append(row.get("rmssd"))
            cond_summary_structured[cond]["sdnn"].append(row.get("sdnn"))

            # ML predictions
            feeling, conf = ml_predict(row)
            if cond not in ml_summary_structured:
                ml_summary_structured[cond] = []
            if feeling is not None:
                ml_summary_structured[cond].append((feeling, conf))

        # Physiological summary text
        cond_summary_text = ""
        for cond, vals in cond_summary_structured.items():
            cond_summary_text += (
                f"- {cond}: "
                f"HR≈{safe_mean(vals['hr']):.1f}, "
                f"alpha_rel≈{safe_mean(vals['alpha_rel']):.3f}, "
                f"beta_rel≈{safe_mean(vals['beta_rel']):.3f}, "
                f"RMSSD≈{safe_mean(vals['rmssd']):.1f}, "
                f"SDNN≈{safe_mean(vals['sdnn']):.1f}\n"
            )

        # ML summary text
        ml_summary_text = ""
        if global_model is None:
            ml_summary_text = "(ML model not loaded — RAG‑only mode)\n"
        else:
            for cond, items in ml_summary_structured.items():
                if len(items) == 0:
                    ml_summary_text += f"- {cond}: (no ML predictions)\n"
                else:
                    best_feeling, best_conf = sorted(items, key=lambda x: x[1], reverse=True)[0]
                    icon = FEELING_ICONS.get(best_feeling, "❓")
                    ml_summary_text += (
                        f"- {cond}: {icon} ML feeling = {best_feeling} (conf={best_conf:.2f})\n"
                    )

        # ----------------------------------------------------------
        # 6) Build context text for LLM
        # ----------------------------------------------------------
        context_text = "\n\n".join([r["text"] for r in results])

        # ----------------------------------------------------------
        # 7) Build LLM prompt (strict RAG + ML hybrid)
        # ----------------------------------------------------------
        prompt = (
            "You are a BIOMEDICAL RESEARCH ASSISTANT specializing in simple, general‑audience explanations "
            "of EEG + ECG data.\n\n"
            "Your job is to compare physiological reactions across conditions in a way that ANY non‑expert "
            "can understand.\n\n"
            "You MUST follow these rules:\n"
            "1. Base ALL statements ONLY on the retrieved context and the physiological summary provided.\n"
            "2. NEVER invent numbers. NEVER hallucinate missing data.\n"
            "3. When data for a condition is missing, clearly state it and use ONLY general physiological expectations.\n"
            "4. Feeling states must be chosen from: relaxed, calm, neutral, tense, stressed.\n"
            "5. Your tone must be simple, clear, and non‑technical.\n\n"
            "Your output MUST contain EXACTLY four sections:\n"
            "------------------------------------------------------------\n"
            "### 1) Blind Spots\n"
            "------------------------------------------------------------\n"
            "### 2) EEG + ECG Reaction Across Conditions\n"
            "------------------------------------------------------------\n"
            "### 3) Feeling‑State Interpretation\n"
            "------------------------------------------------------------\n"
            "### 4) Conclusion\n"
            "------------------------------------------------------------\n\n"
            "PHYSIOLOGICAL SUMMARY (strict RAG):\n"
            f"{cond_summary_text}\n\n"
            "ML FEELING‑STATE SUMMARY:\n"
            f"{ml_summary_text}\n\n"
            "USER QUESTION:\n"
            f"{user_input}\n\n"
            "RETRIEVED CONTEXT:\n"
            f"{context_text}\n\n"
            "Now produce the structured BIOMEDICAL RESEARCH ASSISTANT output."
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        answer = llm.invoke(prompt).content

        # ----------------------------------------------------------
        # 8) Final polished Hybrid Research View rendering
        # ----------------------------------------------------------
        st.markdown(
            f"""
            <div class="summary-card">
                <h3 style="color:#fff;margin-bottom:8px;">🔎 Quick Summary</h3>
                <p style="color:#ccc;">Subject: <strong>{asked_subject or sel_subject}</strong></p>
                <p style="color:#ccc;">Conditions analyzed: <strong>{", ".join(sel_conditions)}</strong></p>
                <p style="color:#ccc;">ML Model: <strong>{"Active" if global_model is not None else "Not Loaded"}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        df_summary = pd.DataFrame([
            {
                "Condition": cond,
                "HR": safe_mean(vals["hr"]),
                "Alpha_rel": safe_mean(vals["alpha_rel"]),
                "Beta_rel": safe_mean(vals["beta_rel"]),
                "RMSSD": safe_mean(vals["rmssd"]),
                "SDNN": safe_mean(vals["sdnn"]),
            }
            for cond, vals in cond_summary_structured.items()
        ])

        if not df_summary.empty:
            st.markdown("#### 📊 Condition‑wise Physiological Summary")
            st.dataframe(df_summary.style.highlight_max(axis=0), use_container_width=True)

        if global_model is not None:
            st.markdown("#### 🔬 ML Feeling‑State Summary")
            for cond, items in ml_summary_structured.items():
                st.markdown(f"**{cond}**")
                if len(items) == 0:
                    st.info("No ML predictions for this condition.")
                else:
                    best_feeling, best_conf = sorted(items, key=lambda x: x[1], reverse=True)[0]
                    icon = FEELING_ICONS.get(best_feeling, "❓")
                    st.markdown(f"{icon} **Feeling:** {best_feeling} (conf={best_conf:.2f})")
                    st.progress(best_conf)

        st.markdown("#### 📘 Structured Neuro‑RAG + ML Report")
        try:
            part1 = answer.split("### 2) EEG + ECG Reaction Across Conditions")[0]
            part2_full = answer.split("### 2) EEG + ECG Reaction Across Conditions")[1]
            part2 = part2_full.split("### 3) Feeling‑State Interpretation")[0]
            part3_full = answer.split("### 3) Feeling‑State Interpretation")[1]
            part3 = part3_full.split("### 4) Conclusion")[0]
            part4 = answer.split("### 4) Conclusion")[1]
        except Exception:
            part1, part2, part3, part4 = answer, "", "", ""

        with st.expander("🔍 Blind Spots", expanded=True):
            st.markdown(part1)

        with st.expander("🧠 EEG + ECG Reaction Across Conditions", expanded=True):
            if part2:
                st.markdown("### 2) EEG + ECG Reaction Across Conditions" + part2)
            else:
                st.markdown("No structured EEG + ECG section parsed.")

        with st.expander("💬 Feeling‑State Interpretation", expanded=True):
            if part3:
                st.markdown("### 3) Feeling‑State Interpretation" + part3)
            else:
                st.markdown("No structured Feeling‑State section parsed.")

        with st.expander("📘 Conclusion", expanded=True):
            if part4:
                st.markdown("### 4) Conclusion" + part4)
            else:
                st.markdown("No structured Conclusion section parsed.")

        render_chat_bubble("assistant", answer)
        st.session_state.rag_chat.append({"role": "assistant", "content": answer})
        st.rerun()
