import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model.train import train_model
from model.predict import composite_score, SIGNAL_NEUTRAL
from model.signals import SIGNALS, FEATURES, CLASS_NAMES, CLASS_COLORS
from model.fitbit import fitbit_to_signals

try:
    from model.grayscale import grayscale_to_signals
    GRAYSCALE_AVAILABLE = True
except ImportError:
    GRAYSCALE_AVAILABLE = False
    def grayscale_to_signals(report):
        neg = report.get("avg_negative", 0) / 60
        neu = report.get("avg_neutral", 0) / 90
        hap = report.get("avg_happy", 0) / 60
        clip = lambda v: round(max(0.0, min(1.0, v)), 4)
        return {
            "facial_negative_load":    clip(neg),
            "facial_flat_affect":      clip(neu),
            "facial_positive_protect": clip(hap),
        }

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="QuietSignals", layout="wide", page_icon="🌿")

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .main { background-color: #f0fdf4; }
    [data-testid="stAppViewContainer"] { background-color: #f0fdf4; }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dcfce7;
    }
    /* Typography */
    h1 { color: #15803d !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }
    h2, h3 { color: #166534 !important; font-weight: 600 !important; }
    p, label, .stCaption { color: #374151 !important; }
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
    }
    [data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #14532d !important; font-weight: 700 !important; }
    /* Expanders */
    details summary {
        color: #15803d !important;
        font-weight: 500 !important;
        border-radius: 10px !important;
    }
    details { border-radius: 10px !important; border: 1px solid #dcfce7 !important; }
    /* Sliders */
    [data-baseweb="slider"] [data-testid="stThumbValue"] { color: #15803d !important; }
    /* Sidebar title */
    [data-testid="stSidebar"] h1 { color: #15803d !important; font-size: 1.3rem !important; }
    [data-testid="stSidebar"] .stCaption { color: #6b7280 !important; }
    /* Divider */
    hr { border-color: #dcfce7 !important; }
    /* Warning */
    [data-testid="stAlert"] { border-radius: 12px !important; }
    /* Radio */
    [data-testid="stRadio"] label { color: #374151 !important; }
</style>
""", unsafe_allow_html=True)

# ── train once per session ─────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return train_model()

clf = get_model()

# ── helpers ───────────────────────────────────────────────────────────────────
RISK_COLOR   = {"Low": "#22c55e", "Moderate": "#f59e0b", "High": "#ef4444"}
RISK_BG      = {"Low": "#f0fdf4", "Moderate": "#fffbeb", "High": "#fef2f2"}
RISK_BORDER  = {"Low": "#86efac", "Moderate": "#fcd34d", "High": "#fca5a5"}

def risk_badge(label):
    color  = RISK_COLOR[label]
    bg     = RISK_BG[label]
    border = RISK_BORDER[label]
    st.markdown(
        f"<div style='background:{bg};color:{color};padding:14px 28px;"
        f"border:2px solid {border};border-radius:50px;"
        f"font-size:1.6rem;font-weight:700;display:inline-block;"
        f"letter-spacing:0.3px;margin-bottom:0.5rem'>"
        f"{label} Risk</div>",
        unsafe_allow_html=True,
    )

def prob_chart(proba):
    fig, ax = plt.subplots(figsize=(4.5, 2.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    colors = CLASS_COLORS
    bars = ax.bar(CLASS_NAMES, proba, color=colors, width=0.45,
                  edgecolor="none", zorder=2)
    for bar, p in zip(bars, proba):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{p:.0%}", ha="center", color="#374151", fontsize=10, fontweight="600")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Probability", color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=9)
    ax.yaxis.grid(True, color="#f1f5f9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=1.2)
    return fig

def contribution_chart(sv):
    labels, values, colors = [], [], []
    behavioral_color = "#4ade80"
    facial_color     = "#86efac"
    for f in FEATURES:
        v = sv.get(f, SIGNAL_NEUTRAL)
        contrib = (1 - v if SIGNALS[f]["inverse"] else v) * SIGNALS[f]["weight"]
        labels.append(SIGNALS[f]["label"])
        values.append(contrib)
        colors.append(facial_color if f.startswith("facial") else behavioral_color)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.barh(labels, values, color=colors, edgecolor="none", height=0.55, zorder=2)
    ax.set_xlabel("Weighted contribution", color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#6b7280", labelsize=8)
    ax.xaxis.grid(True, color="#f1f5f9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    legend_patches = [
        mpatches.Patch(color=behavioral_color, label="Behavioral"),
        mpatches.Patch(color=facial_color,     label="Facial"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, frameon=False,
              labelcolor="#6b7280", loc="lower right")
    fig.tight_layout(pad=1.2)
    return fig

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🌿 QuietSignals")
st.sidebar.caption("Nurse burnout risk scorer")
st.sidebar.divider()

input_mode = st.sidebar.radio(
    "Signal source",
    ["Manual", "Fitbit", "Manual + Facial"],
    index=0,
)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("# QuietSignals")
st.caption("Real-time burnout risk assessment from behavioral and physiological signals.")

if not GRAYSCALE_AVAILABLE:
    st.warning(
        "deepface / opencv not installed — facial signals use built-in normalizer. "
        "Run `pip install deepface opencv-python` to enable live analysis.",
        icon="⚠️",
    )

st.divider()

left, right = st.columns([1, 1], gap="large")

sv = {}

# ── LEFT: inputs ──────────────────────────────────────────────────────────────
with left:

    if input_mode == "Manual":
        st.markdown("### Behavioral Signals")
        behavioral = [f for f in FEATURES if not f.startswith("facial")]
        for f in behavioral:
            sv[f] = st.slider(
                SIGNALS[f]["label"],
                0.0, 1.0,
                value=0.5, step=0.01, key=f
            )
        st.caption("Facial signals not provided — neutral (0.5) will be used.")

    elif input_mode == "Fitbit":
        st.markdown("### Fitbit Sensor Values")
        hrv       = st.slider("HRV RMSSD (ms)", 10.0, 55.0, 35.0, step=0.5)
        step_reg  = st.slider("Step Regularity", 0.1, 0.8, 0.5, step=0.01)
        sleep_eff = st.slider("Sleep Efficiency (%)", 40.0, 98.0, 80.0, step=0.5)
        sleep_hrs = st.slider("Sleep Duration (hrs)", 2.0, 9.0, 6.5, step=0.1)
        sed_bouts = st.slider("Sedentary Bouts", 0, 14, 5)
        sov       = st.slider("Sleep Onset Variability (hrs)", 0.1, 2.0, 0.6, step=0.05)
        n_awk     = st.slider("Night Awakenings", 0, 12, 3)

        fitbit_row = {
            "hrv_rmssd": hrv,
            "step_regularity": step_reg,
            "sleep_efficiency": sleep_eff,
            "sleep_duration_hrs": sleep_hrs,
            "sedentary_bouts": sed_bouts,
            "sleep_onset_variability": sov,
            "n_awakenings": n_awk,
        }
        sv.update(fitbit_to_signals(fitbit_row))
        st.caption("Facial signals not provided — neutral (0.5) will be used.")

    else:  # Manual + Facial
        st.markdown("### Behavioral Signals")
        behavioral = [f for f in FEATURES if not f.startswith("facial")]
        for f in behavioral:
            sv[f] = st.slider(
                SIGNALS[f]["label"],
                0.0, 1.0,
                value=0.5, step=0.01, key=f
            )
        st.markdown("### Facial Signals")
        st.caption("Enter values as if returned by grayscale DeepFace analysis.")
        avg_negative = st.slider("Avg Negative Emotion Load (%)", 0.0, 100.0, 20.0, step=1.0)
        avg_neutral  = st.slider("Avg Neutral Expression (%)",    0.0, 100.0, 50.0, step=1.0)
        avg_happy    = st.slider("Avg Happy Expression (%)",      0.0, 100.0, 20.0, step=1.0)

        grayscale_report = {
            "avg_negative": avg_negative,
            "avg_neutral":  avg_neutral,
            "avg_happy":    avg_happy,
        }
        sv.update(grayscale_to_signals(grayscale_report))

    for f in FEATURES:
        sv.setdefault(f, SIGNAL_NEUTRAL)

# ── RIGHT: results ─────────────────────────────────────────────────────────────
with right:
    score = composite_score(sv)
    x     = np.array([[sv[f] for f in FEATURES]])
    proba = clf.predict_proba(x)[0]
    label = CLASS_NAMES[int(np.argmax(proba))]

    st.markdown("### Result")
    risk_badge(label)

    st.metric("Composite score", f"{score:.4f}", help="0 = no burnout signal, 1 = maximum")

    st.pyplot(prob_chart(proba))

    with st.expander("Signal contributions"):
        st.pyplot(contribution_chart(sv))

    with st.expander("Raw signal values"):
        import pandas as pd
        rows = []
        for f in FEATURES:
            v = sv[f]
            rows.append({
                "Signal":   SIGNALS[f]["label"],
                "Value":    f"{v:.3f}",
                "Inverse":  "yes" if SIGNALS[f]["inverse"] else "no",
                "Weight":   SIGNALS[f]["weight"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
