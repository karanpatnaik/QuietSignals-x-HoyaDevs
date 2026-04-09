import sys
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="QuietSignals", layout="wide", page_icon="🌿")

# ── color palette ────────────────────────────────────────────────────────────
C = {
    "bg":       "#f0fdf4",
    "panel":    "#ffffff",
    "teal":     "#0d9488",
    "green":    "#16a34a",
    "amber":    "#d97706",
    "red":      "#dc2626",
    "indigo":   "#4f46e5",
    "muted":    "#64748b",
    "border":   "#e2e8f0",
    "chart_bg": "#f8fafc",
    "low_fg":   "#15803d",  "low_bg":  "#f0fdf4",  "low_bd":  "#86efac",
    "mod_fg":   "#b45309",  "mod_bg":  "#fffbeb",  "mod_bd":  "#fcd34d",
    "hi_fg":    "#b91c1c",  "hi_bg":   "#fef2f2",  "hi_bd":   "#fca5a5",
}

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main,[data-testid="stAppViewContainer"]{background:#f0fdf4}
    [data-testid="stSidebar"]{background:#fff;border-right:2px solid #dcfce7}
    h1{color:#0d9488!important;font-weight:800!important}
    h2,h3{color:#115e59!important;font-weight:600!important}
    p,label{color:#374151!important}
    [data-testid="metric-container"]{
        background:#fff;border:1px solid #ccfbf1;
        border-radius:14px;padding:1rem 1.2rem}
    [data-testid="stMetricLabel"]{color:#64748b!important;font-size:.8rem!important}
    [data-testid="stMetricValue"]{color:#0d9488!important;font-weight:700!important}
    details{border-radius:10px!important;border:1px solid #ccfbf1!important}
    details summary{color:#0d9488!important;font-weight:500!important}
    hr{border-color:#ccfbf1!important}
    [data-testid="stAlert"]{border-radius:12px!important}
    .stTabs [data-baseweb="tab-list"]{background:#f0fdf4;border-radius:12px;padding:4px}
    .stTabs [data-baseweb="tab"]{border-radius:8px;padding:8px 16px}
    .stTabs [aria-selected="true"]{background:#0d9488!important;color:#fff!important}
</style>
""", unsafe_allow_html=True)

# ── model (cached) ───────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return train_model()

clf = get_model()


# ── score → label  (fixes Low-range issue — uses thresholds, not argmax) ────
def score_to_label(score: float) -> str:
    if score < 0.33:
        return "Low"
    elif score < 0.55:
        return "Moderate"
    return "High"


RISK_COLOR  = {"Low": C["low_fg"], "Moderate": C["mod_fg"], "High": C["hi_fg"]}
RISK_BG     = {"Low": C["low_bg"], "Moderate": C["mod_bg"], "High": C["hi_bg"]}
RISK_BORDER = {"Low": C["low_bd"], "Moderate": C["mod_bd"], "High": C["hi_bd"]}


# ── mock nurse profile database ──────────────────────────────────────────────
@st.cache_data
def get_nurse_profiles() -> pd.DataFrame:
    np.random.seed(99)
    names = [
        "Sarah Chen", "Marcus Johnson", "Priya Patel", "James Wilson",
        "Maria Garcia", "David Kim", "Lisa Thompson", "Robert Davis",
        "Emily Anderson", "Michael Brown", "Jennifer Lee", "Chris Martinez",
        "Amanda Taylor", "Jonathan Jackson", "Samantha White", "Daniel Harris",
        "Ashley Clark", "Matthew Lewis", "Stephanie Robinson", "Ryan Walker",
    ]
    depts  = ["ICU", "ER", "Med-Surg", "Oncology", "Pediatrics", "NICU", "OR"]
    shifts = ["Day", "Night", "Rotating"]

    profiles = []
    for i, name in enumerate(names):
        dept      = np.random.choice(depts)
        shift     = np.random.choice(shifts, p=[0.5, 0.3, 0.2])
        years_exp = int(np.random.randint(1, 26))

        base = float(np.random.beta(1.8, 3.0))
        history: list[float] = [base]
        for _ in range(7):
            history.append(float(np.clip(history[-1] + np.random.normal(0, 0.05), 0.05, 0.95)))
        history = [round(h, 3) for h in history]
        cs = history[-1]

        profiles.append({
            "id": f"RN-{1000 + i}",
            "name": name,
            "department": dept,
            "shift": shift,
            "years_exp": years_exp,
            "history": history,
            "current_score": cs,
            "risk_level": score_to_label(cs),
        })
    return pd.DataFrame(profiles)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_base(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C["panel"])
    ax.set_facecolor(C["chart_bg"])
    return fig, ax


def prob_chart(proba):
    """Bar chart of class probabilities with % labels and axis titles."""
    fig, ax = _fig_base(5, 3)
    palette = [C["green"], C["amber"], C["red"]]
    bars = ax.bar(CLASS_NAMES, proba, color=palette, width=0.5,
                  edgecolor="white", linewidth=1.5, zorder=2)
    for bar, p in zip(bars, proba):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{p:.0%}", ha="center", va="bottom",
                color="#374151", fontsize=11, fontweight="700")
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Probability", color=C["muted"], fontsize=9)
    ax.set_xlabel("Burnout Risk Class", color=C["muted"], fontsize=9)
    ax.tick_params(colors=C["muted"], labelsize=10)
    ax.yaxis.grid(True, color=C["border"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Class Probability Distribution", fontsize=10,
                 color="#374151", pad=8, fontweight="600")
    fig.tight_layout(pad=1.2)
    return fig


def radar_chart(sv: dict):
    """Spider/radar chart showing all 10 signal values with risk-zone shading."""
    short = {
        "gsr": "GSR", "task_switch": "Task Switch", "voice_monotony": "Voice Mon.",
        "gait_irregularity": "Gait", "patient_rel": "Patient Rel.",
        "color_chaos": "Sleep Chaos", "tiktok_burnout": "TikTok",
        "facial_negative_load": "Neg. Load", "facial_flat_affect": "Flat Affect",
        "facial_positive_protect": "Pos. Affect",
    }
    N = len(FEATURES)
    vals   = [sv.get(f, SIGNAL_NEUTRAL) for f in FEATURES]
    labels = [short[f] for f in FEATURES]

    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    vals   = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(5, 4.5), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor(C["panel"])
    ax.set_facecolor(C["chart_bg"])

    # Risk-zone fills
    zone_angles = np.linspace(0, 2 * np.pi, 200)
    ax.fill(zone_angles, [1.0] * 200,  color=C["red"],   alpha=0.04)
    ax.fill(zone_angles, [0.55] * 200, color=C["amber"], alpha=0.07)
    ax.fill(zone_angles, [0.33] * 200, color=C["green"], alpha=0.09)

    ax.plot(angles, vals, "o-", linewidth=2.2, color=C["teal"], markersize=5)
    ax.fill(angles, vals, alpha=0.22, color=C["teal"])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=7.5, color="#374151")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=6, color=C["muted"])
    ax.grid(color=C["border"], linewidth=0.6)
    ax.set_title("Signal Radar Profile", fontsize=10, color="#374151",
                 pad=16, fontweight="600")
    fig.tight_layout(pad=0.5)
    return fig


def contribution_chart(sv: dict):
    """Horizontal bar chart of weighted signal contributions, color-coded by magnitude."""
    labels, values, colors = [], [], []
    for f in FEATURES:
        v = sv.get(f, SIGNAL_NEUTRAL)
        contrib = (1 - v if SIGNALS[f]["inverse"] else v) * SIGNALS[f]["weight"]
        labels.append(SIGNALS[f]["label"])
        values.append(contrib)
        if f.startswith("facial"):
            colors.append(C["indigo"])
        elif contrib > 0.08:
            colors.append(C["red"])
        elif contrib > 0.04:
            colors.append(C["amber"])
        else:
            colors.append(C["green"])

    fig, ax = _fig_base(5.5, 4.2)
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.6, zorder=2)
    for bar, v in zip(bars, values):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8, color=C["muted"])
    ax.set_xlabel("Weighted Contribution to Burnout Score", color=C["muted"], fontsize=9)
    ax.set_ylabel("Signal", color=C["muted"], fontsize=9)
    ax.tick_params(colors="#374151", labelsize=8)
    ax.xaxis.grid(True, color=C["border"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.invert_yaxis()

    patches = [
        mpatches.Patch(color=C["red"],    label="High contribution (>0.08)"),
        mpatches.Patch(color=C["amber"],  label="Moderate (0.04–0.08)"),
        mpatches.Patch(color=C["green"],  label="Low (<0.04)"),
        mpatches.Patch(color=C["indigo"], label="Facial signal"),
    ]
    ax.legend(handles=patches, fontsize=7.5, frameon=False,
              labelcolor=C["muted"], loc="lower right")
    ax.set_title("Signal Contributions to Burnout Score", fontsize=10,
                 color="#374151", pad=8, fontweight="600")
    fig.tight_layout(pad=1.2)
    return fig


def trend_chart(weeks: list, scores: list, forecast: list | None = None):
    """Line chart of historical burnout scores + optional 4-week forecast."""
    fig, ax = _fig_base(6.5, 3)
    ax.plot(weeks, scores, "o-", color=C["teal"], linewidth=2.5, markersize=6,
            label="Burnout Score", zorder=3)
    ax.fill_between(weeks, scores, alpha=0.12, color=C["teal"])

    if forecast:
        fw = [weeks[-1] + i for i in range(1, len(forecast) + 1)]
        ax.plot(fw, forecast, "o--", color=C["indigo"], linewidth=2,
                markersize=5, label="4-Week Forecast", zorder=3, alpha=0.85)
        ax.fill_between(fw, forecast, alpha=0.08, color=C["indigo"])

    ax.axhline(0.33, color=C["amber"], linestyle="--", linewidth=1.2,
               alpha=0.8, label="Moderate threshold")
    ax.axhline(0.55, color=C["red"],   linestyle="--", linewidth=1.2,
               alpha=0.8, label="High threshold")

    ax.set_ylabel("Burnout Score", color=C["muted"], fontsize=9)
    ax.set_xlabel("Week", color=C["muted"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.tick_params(colors=C["muted"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color=C["border"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, labelcolor=C["muted"])
    ax.set_title("Burnout Score Trend & Forecast", fontsize=10,
                 color="#374151", pad=8, fontweight="600")
    fig.tight_layout(pad=1.2)
    return fig


def dept_bar_chart(dept_stats: pd.DataFrame):
    """Bar chart of average burnout score per department with threshold lines."""
    depts  = dept_stats["department"].tolist()
    scores = dept_stats["avg_score"].tolist()
    colors = [C["red"] if s >= 0.55 else (C["amber"] if s >= 0.33 else C["green"])
              for s in scores]

    fig, ax = _fig_base(7, 3.8)
    bars = ax.bar(depts, scores, color=colors, width=0.55,
                  edgecolor="white", linewidth=1.2, zorder=2)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{s:.3f}", ha="center", fontsize=9, fontweight="600", color="#374151")
    ax.axhline(0.33, color=C["amber"], linestyle="--", linewidth=1.3,
               alpha=0.85, label="Moderate threshold (0.33)")
    ax.axhline(0.55, color=C["red"],   linestyle="--", linewidth=1.3,
               alpha=0.85, label="High threshold (0.55)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Avg Burnout Score", color=C["muted"], fontsize=9)
    ax.set_xlabel("Department", color=C["muted"], fontsize=9)
    ax.tick_params(colors="#374151", labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color=C["border"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False, labelcolor=C["muted"])
    ax.set_title("Average Burnout Score by Department", fontsize=11,
                 color="#374151", pad=8, fontweight="600")
    fig.tight_layout(pad=1.2)
    return fig


def hrv_hr_24h_chart(hours, hrv_vals, hr_vals):
    """Dual-panel 24-hour physiological trend for Fitbit live feed."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    fig.patch.set_facecolor(C["panel"])

    for ax in (ax1, ax2):
        ax.set_facecolor(C["chart_bg"])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.yaxis.grid(True, color=C["border"], linewidth=0.6)
        ax.tick_params(colors=C["muted"], labelsize=8)

    ax1.plot(hours, hrv_vals, color=C["teal"], linewidth=2.5, label="HRV RMSSD")
    ax1.fill_between(hours, hrv_vals, alpha=0.15, color=C["teal"])
    ax1.axhline(35, color=C["amber"], linestyle="--", linewidth=1.2,
                alpha=0.8, label="Threshold: 35 ms")
    ax1.set_ylabel("HRV RMSSD (ms)", color=C["muted"], fontsize=9)
    ax1.legend(fontsize=8, frameon=False, labelcolor=C["muted"])
    ax1.set_title("24-Hour Physiological Trend", fontsize=10,
                  color="#374151", pad=8, fontweight="600")

    ax2.plot(hours, hr_vals, color=C["red"], linewidth=2.5, label="Resting HR")
    ax2.fill_between(hours, hr_vals, alpha=0.12, color=C["red"])
    ax2.axhline(75, color=C["amber"], linestyle="--", linewidth=1.2,
                alpha=0.8, label="Threshold: 75 bpm")
    ax2.set_ylabel("Heart Rate (bpm)", color=C["muted"], fontsize=9)
    ax2.set_xlabel("Hour of Day (0–23)", color=C["muted"], fontsize=9)
    ax2.legend(fontsize=8, frameon=False, labelcolor=C["muted"])

    fig.tight_layout(pad=1.3)
    return fig


def stroop_result_chart(correct: int, total: int, times: list):
    """Pie + response-time line for the Color Chaos Exercise results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    fig.patch.set_facecolor(C["panel"])
    ax1.set_facecolor(C["panel"])
    ax2.set_facecolor(C["chart_bg"])

    ax1.pie([correct, total - correct],
            labels=["Correct", "Incorrect"],
            colors=[C["green"], C["red"]],
            autopct="%1.0f%%", startangle=90,
            textprops={"color": "#374151", "fontsize": 10})
    ax1.set_title("Accuracy Breakdown", fontsize=10, color="#374151", fontweight="600")

    if times:
        avg_rt = np.mean(times)
        ax2.plot(range(1, len(times) + 1), times, "o-", color=C["teal"],
                 linewidth=2, markersize=6)
        ax2.axhline(avg_rt, color=C["amber"], linestyle="--", linewidth=1.2,
                    alpha=0.8, label=f"Avg {avg_rt:.1f}s")
        ax2.set_xlabel("Trial", color=C["muted"], fontsize=9)
        ax2.set_ylabel("Response Time (s)", color=C["muted"], fontsize=9)
        ax2.tick_params(colors=C["muted"], labelsize=8)
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.yaxis.grid(True, color=C["border"], linewidth=0.5)
        ax2.legend(fontsize=8, frameon=False, labelcolor=C["muted"])
        ax2.set_title("Response Time per Trial", fontsize=10,
                      color="#374151", fontweight="600")

    fig.tight_layout(pad=1.5)
    return fig


# ── clinical helpers ──────────────────────────────────────────────────────────

def risk_badge(label: str):
    color  = RISK_COLOR[label]
    bg     = RISK_BG[label]
    border = RISK_BORDER[label]
    icon   = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}[label]
    st.markdown(
        f"<div style='background:{bg};color:{color};padding:14px 28px;"
        f"border:2px solid {border};border-radius:50px;"
        f"font-size:1.6rem;font-weight:700;display:inline-block;"
        f"letter-spacing:0.3px;margin-bottom:0.5rem'>"
        f"{icon} {label} Risk</div>",
        unsafe_allow_html=True,
    )


def warning_flags(sv: dict) -> list:
    """Return list of (icon, label, signal_key, value) for elevated signals."""
    flags = []
    high_thresh = {
        "gsr":                 ("High stress arousal",       0.65),
        "task_switch":         ("Cognitive overload",         0.65),
        "voice_monotony":      ("Vocal disengagement",        0.60),
        "gait_irregularity":   ("Movement irregularity",      0.60),
        "color_chaos":         ("Sleep schedule chaos",       0.65),
        "tiktok_burnout":      ("Sleep disruption",           0.65),
        "facial_negative_load":("Negative emotional load",    0.55),
        "facial_flat_affect":  ("Flat affect / depersonal.",  0.55),
    }
    low_thresh = {
        "patient_rel":         ("Poor patient relationship quality", 0.35),
    }
    for sig, (lbl, thresh) in high_thresh.items():
        if sv.get(sig, SIGNAL_NEUTRAL) >= thresh:
            flags.append(("🔴", lbl, sig, sv.get(sig, SIGNAL_NEUTRAL)))
    for sig, (lbl, thresh) in low_thresh.items():
        if sv.get(sig, SIGNAL_NEUTRAL) <= thresh:
            flags.append(("🟠", lbl, sig, sv.get(sig, SIGNAL_NEUTRAL)))
    return flags


def ai_suggestions(sv: dict, label: str) -> list:
    """Rule-based clinical suggestions derived from signal values."""
    tips = []
    if sv.get("gsr", 0) > 0.65:
        tips.append("Practice 4-7-8 guided breathing between patient interactions to dampen sympathetic activation.")
    if sv.get("task_switch", 0) > 0.65:
        tips.append("Discuss task batching with charge nurse — grouped documentation reduces cognitive interrupt load.")
    if sv.get("voice_monotony", 0) > 0.60:
        tips.append("Schedule brief daily peer check-ins. Vocal social engagement reduces emotional disengagement.")
    if sv.get("gait_irregularity", 0) > 0.60:
        tips.append("Check hydration levels and footwear fit. Irregular gait may indicate fatigue or musculoskeletal strain.")
    if sv.get("patient_rel", SIGNAL_NEUTRAL) < 0.35:
        tips.append("Patient relationship quality is declining — consider EAP referral for compassion fatigue support.")
    if sv.get("color_chaos", 0) > 0.65:
        tips.append("Sleep schedule irregularity is high. Consistent wake/sleep times and shift rotation review recommended.")
    if sv.get("tiktok_burnout", 0) > 0.65:
        tips.append("High sleep disruption detected. Reduce screen time 1 hr before sleep; consider a sleep diary.")
    if sv.get("facial_negative_load", 0) > 0.55:
        tips.append("Elevated negative emotional expression detected. Mental health check-in or EAP consultation recommended.")
    if sv.get("facial_flat_affect", 0) > 0.55:
        tips.append("Reduced emotional expressivity may indicate depersonalization — consider peer support or engagement program.")
    if label == "High":
        tips.insert(0, "PRIORITY: Notify charge nurse / unit manager for an immediate well-being check-in.")
        if not tips[1:]:
            tips.append("Multiple elevated signals. Recommend temporary workload reduction and structured recovery plan.")
    elif label == "Moderate" and not tips:
        tips.append("Monitor weekly. Focus on sleep consistency, adequate recovery time, and peer connection.")
    elif label == "Low" and not tips:
        tips.append("All signals within healthy range. Continue current self-care routines and check in monthly.")
    return tips[:6]


def _flag_card(icon, text, color="#fef2f2", border="#fca5a5", text_color="#991b1b"):
    st.markdown(
        f"<div style='background:{color};border:1px solid {border};"
        f"border-radius:8px;padding:6px 12px;margin:4px 0;"
        f"font-size:.85rem;color:{text_color}'>{icon} {text}</div>",
        unsafe_allow_html=True,
    )


def _tip_card(tip):
    is_priority = tip.startswith("PRIORITY") or tip.startswith("ATTENTION")
    bg, bd, tc = (
        ("#fef2f2", "#fca5a5", "#991b1b") if is_priority
        else ("#f0fdf4", "#86efac", "#166534")
    )
    icon = "⚡" if is_priority else "💡"
    st.markdown(
        f"<div style='background:{bg};border:1px solid {bd};"
        f"border-radius:8px;padding:8px 12px;margin:5px 0;"
        f"font-size:.85rem;color:{tc}'>{icon} {tip}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — nurse identification + input mode
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🌿 QuietSignals")
st.sidebar.caption("Nurse Burnout Intelligence Platform")
st.sidebar.divider()

st.sidebar.markdown("#### Nurse Identification")
nurse_name  = st.sidebar.text_input("Full Name",        placeholder="e.g. Sarah Chen")
nurse_id    = st.sidebar.text_input("Employee / RN ID", placeholder="e.g. RN-1042")
nurse_dept  = st.sidebar.selectbox(
    "Department", ["—", "ICU", "ER", "Med-Surg", "Oncology", "Pediatrics", "NICU", "OR", "Other"])
nurse_shift = st.sidebar.selectbox("Shift", ["—", "Day", "Night", "Rotating"])

identified = bool(nurse_name and nurse_id and nurse_dept != "—")
if identified:
    st.sidebar.success(f"✓ **{nurse_name}**  \n{nurse_dept} · {nurse_shift} Shift")
else:
    st.sidebar.info("Enter name + ID to enable personalized tracking.")

st.sidebar.divider()
input_mode = st.sidebar.radio(
    "Signal source",
    ["Manual", "Fitbit", "Manual + Facial"],
    index=0,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown("# 🌿 QuietSignals")
    st.caption("Real-time burnout risk · Behavioral & physiological signals · Powered by ML")
with hcol2:
    if identified:
        st.markdown(
            f"<div style='text-align:right;padding:8px 12px;background:{C['low_bg']};"
            f"border:1px solid {C['low_bd']};border-radius:10px;"
            f"font-size:.85rem;color:{C['low_fg']}'>"
            f"<strong>{nurse_name}</strong><br>{nurse_id} · {nurse_dept}</div>",
            unsafe_allow_html=True,
        )

if not GRAYSCALE_AVAILABLE:
    st.warning(
        "deepface / opencv not installed — facial signals use built-in normalizer. "
        "Run `pip install deepface opencv-python` to enable live webcam analysis.",
        icon="⚠️",
    )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏥 Assessment",
    "🏢 Department Overview",
    "👥 Nurse Profiles & Forecast",
    "📡 Fitbit Live Feed",
    "🎨 Color Chaos Exercise",
])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 · INDIVIDUAL ASSESSMENT
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    sv: dict = {}
    left, right = st.columns([1, 1.1], gap="large")

    # ── inputs ────────────────────────────────────────────────────────────────
    with left:
        if input_mode == "Manual":
            st.markdown("### Behavioral Signals")
            for f in [x for x in FEATURES if not x.startswith("facial")]:
                sv[f] = st.slider(SIGNALS[f]["label"], 0.0, 1.0, 0.5, 0.01, key=f"m_{f}")
            st.caption("ℹ Facial signals defaulted to neutral (0.5).")

        elif input_mode == "Fitbit":
            st.markdown("### Fitbit Sensor Values")
            hrv       = st.slider("HRV RMSSD (ms)",                10.0, 55.0, 35.0, 0.5)
            step_reg  = st.slider("Step Regularity",                 0.1,  0.8,  0.5, 0.01)
            sleep_eff = st.slider("Sleep Efficiency (%)",           40.0, 98.0, 80.0, 0.5)
            sleep_hrs = st.slider("Sleep Duration (hrs)",            2.0,  9.0,  6.5, 0.1)
            sed_bouts = st.slider("Sedentary Bouts",                   0,   14,    5)
            sov       = st.slider("Sleep Onset Variability (hrs)",   0.1,  2.0,  0.6, 0.05)
            n_awk     = st.slider("Night Awakenings",                  0,   12,    3)
            sv.update(fitbit_to_signals({
                "hrv_rmssd": hrv, "step_regularity": step_reg,
                "sleep_efficiency": sleep_eff, "sleep_duration_hrs": sleep_hrs,
                "sedentary_bouts": sed_bouts, "sleep_onset_variability": sov,
                "n_awakenings": n_awk,
            }))
            st.caption("ℹ Facial signals defaulted to neutral (0.5).")

        else:  # Manual + Facial
            st.markdown("### Behavioral Signals")
            for f in [x for x in FEATURES if not x.startswith("facial")]:
                sv[f] = st.slider(SIGNALS[f]["label"], 0.0, 1.0, 0.5, 0.01, key=f"mf_{f}")
            st.markdown("### Facial Signals")
            st.caption("Values as returned by DeepFace grayscale analysis.")
            avg_neg = st.slider("Avg Negative Emotion Load (%)", 0.0, 100.0, 20.0, 1.0)
            avg_neu = st.slider("Avg Neutral Expression (%)",    0.0, 100.0, 50.0, 1.0)
            avg_hap = st.slider("Avg Happy Expression (%)",      0.0, 100.0, 20.0, 1.0)
            sv.update(grayscale_to_signals(
                {"avg_negative": avg_neg, "avg_neutral": avg_neu, "avg_happy": avg_hap}))

    for f in FEATURES:
        sv.setdefault(f, SIGNAL_NEUTRAL)

    # ── results ───────────────────────────────────────────────────────────────
    with right:
        score = composite_score(sv)
        label = score_to_label(score)          # ← score-based, not argmax
        x     = np.array([[sv[f] for f in FEATURES]])
        proba = clf.predict_proba(x)[0]

        st.markdown("### Assessment Result")
        risk_badge(label)

        m1, m2, m3 = st.columns(3)
        m1.metric("Composite Score",  f"{score:.4f}",
                  help="0 = no burnout signal · 1 = maximum")
        m2.metric("ML Confidence",    f"{max(proba):.0%}",
                  help="Classifier confidence in predicted class")
        m3.metric("Signals Analyzed", str(len(FEATURES)))

        # Warning flags
        flags = warning_flags(sv)
        if flags:
            with st.expander(f"⚠️ Warning Flags — {len(flags)} signal(s) elevated",
                             expanded=True):
                for icon, lbl, _sig, val in flags:
                    _flag_card(icon, f"<strong>{lbl}</strong> &nbsp;— value: {val:.3f}")
        else:
            st.success("✓ No signals are in critically elevated range.", icon="✅")

        # AI suggestions
        tips = ai_suggestions(sv, label)
        if tips:
            with st.expander("💡 AI Clinical Suggestions",
                             expanded=(label != "Low")):
                for tip in tips:
                    _tip_card(tip)

        # Charts: probability bar + radar side-by-side
        cc1, cc2 = st.columns(2)
        with cc1:
            st.pyplot(prob_chart(proba))
        with cc2:
            st.pyplot(radar_chart(sv))

        with st.expander("📊 Signal Contributions"):
            st.pyplot(contribution_chart(sv))

        with st.expander("📋 Raw Signal Values"):
            rows = []
            for f in FEATURES:
                v      = sv[f]
                contrib = (1 - v if SIGNALS[f]["inverse"] else v) * SIGNALS[f]["weight"]
                status  = "🔴" if contrib > 0.08 else ("🟡" if contrib > 0.04 else "🟢")
                rows.append({
                    "Signal":       SIGNALS[f]["label"],
                    "Value":        f"{v:.3f}",
                    "Contribution": f"{contrib:.4f}",
                    "Status":       status,
                    "Inverse":      "yes" if SIGNALS[f]["inverse"] else "no",
                    "Weight":       SIGNALS[f]["weight"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 · DEPARTMENT / HOSPITAL OVERVIEW
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Department & Hospital Overview")
    profiles = get_nurse_profiles()

    dept_stats = (
        profiles.groupby("department")
        .agg(
            avg_score   =("current_score", "mean"),
            n_nurses    =("id", "count"),
            n_high      =("risk_level", lambda x: (x == "High").sum()),
            n_moderate  =("risk_level", lambda x: (x == "Moderate").sum()),
            n_low       =("risk_level", lambda x: (x == "Low").sum()),
        )
        .reset_index()
    )
    dept_stats["avg_score"] = dept_stats["avg_score"].round(3)

    total  = len(profiles)
    n_high = (profiles["risk_level"] == "High").sum()
    n_mod  = (profiles["risk_level"] == "Moderate").sum()
    n_low  = (profiles["risk_level"] == "Low").sum()
    avg_s  = profiles["current_score"].mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Nurses",     total)
    m2.metric("Hospital Avg Score", f"{avg_s:.3f}")
    m3.metric("🔴 High Risk",     n_high,  f"{n_high/total:.0%} of staff")
    m4.metric("🟡 Moderate Risk", n_mod,   f"{n_mod/total:.0%} of staff")
    m5.metric("🟢 Low Risk",      n_low,   f"{n_low/total:.0%} of staff")

    st.divider()

    ch_col, tbl_col = st.columns([1.4, 1])
    with ch_col:
        st.pyplot(dept_bar_chart(dept_stats))
    with tbl_col:
        st.markdown("#### Department Risk Summary")
        display = dept_stats.rename(columns={
            "department": "Dept", "n_nurses": "Nurses",
            "avg_score": "Avg Score", "n_high": "🔴 High",
            "n_moderate": "🟡 Mod", "n_low": "🟢 Low",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Flagged Nurses — Requires Attention")
    flagged = (
        profiles[profiles["risk_level"].isin(["High", "Moderate"])]
        .sort_values("current_score", ascending=False)
    )
    st.dataframe(
        flagged[["id", "name", "department", "shift", "current_score", "risk_level"]].rename(
            columns={"id": "ID", "name": "Name", "department": "Dept",
                     "shift": "Shift", "current_score": "Score", "risk_level": "Risk"}
        ),
        use_container_width=True, hide_index=True,
    )

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 · NURSE PROFILES & BURNOUT FORECAST
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Nurse Profiles & Burnout Forecasting")
    profiles = get_nurse_profiles()

    sel_col, info_col = st.columns([1, 2])
    with sel_col:
        selected = st.selectbox("Select Nurse", profiles["name"].tolist())

    row = profiles[profiles["name"] == selected].iloc[0]

    with info_col:
        rfg = RISK_COLOR[row["risk_level"]]
        rbg = RISK_BG[row["risk_level"]]
        rbd = RISK_BORDER[row["risk_level"]]
        st.markdown(
            f"<div style='background:{rbg};border:2px solid {rbd};"
            f"border-radius:14px;padding:14px 20px'>"
            f"<strong style='color:{rfg};font-size:1.1rem'>{row['name']}</strong><br>"
            f"<span style='color:{C['muted']};font-size:.85rem'>"
            f"ID: {row['id']} &nbsp;·&nbsp; {row['department']} &nbsp;·&nbsp; {row['shift']} Shift "
            f"&nbsp;·&nbsp; {row['years_exp']} yrs experience</span><br>"
            f"<span style='color:{rfg};font-weight:600'>"
            f"● {row['risk_level']} Risk &nbsp;|&nbsp; Score: {row['current_score']:.3f}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    history = list(row["history"])
    weeks   = list(range(1, len(history) + 1))

    # Deterministic linear forecast
    np.random.seed(abs(hash(selected)) % (2 ** 31))
    coeffs   = np.polyfit(weeks, history, 1)
    fw       = list(range(len(history) + 1, len(history) + 5))
    forecast = [
        float(np.clip(np.polyval(coeffs, w) + np.random.normal(0, 0.03), 0.05, 0.95))
        for w in fw
    ]
    forecast = [round(f, 3) for f in forecast]

    ch_col2, st_col2 = st.columns([1.5, 1])
    with ch_col2:
        st.pyplot(trend_chart(weeks, history, forecast))
    with st_col2:
        st.markdown("#### 8-Week Statistics")
        trend_dir = "↑ Worsening" if coeffs[0] > 0.01 else ("↓ Improving" if coeffs[0] < -0.01 else "→ Stable")
        st.metric("Current Score",    f"{history[-1]:.3f}")
        st.metric("8-Week Average",   f"{np.mean(history):.3f}")
        st.metric("Score Trend",      f"{trend_dir}  ({abs(coeffs[0]):.3f}/wk)")
        st.metric("4-Week Forecast",  f"{forecast[-1]:.3f}",
                  delta=f"Predicted: {score_to_label(forecast[-1])}")

    st.divider()

    st.markdown("#### Department Peer Comparison")
    peers    = profiles[profiles["department"] == row["department"]]
    dept_avg = peers["current_score"].mean()
    pct      = (peers["current_score"] < row["current_score"]).mean()

    pc1, pc2 = st.columns(2)
    pc1.metric(f"{row['department']} Dept Avg",
               f"{dept_avg:.3f}",
               delta=f"This nurse {'above' if row['current_score'] > dept_avg else 'below'} avg")
    pc2.metric("Dept Burnout Percentile",
               f"{pct:.0%}",
               help="% of dept peers with a lower burnout score")

    with st.expander(f"All {row['department']} nurses ({len(peers)})"):
        st.dataframe(
            peers[["name", "shift", "years_exp", "current_score", "risk_level"]].rename(
                columns={"name": "Name", "shift": "Shift", "years_exp": "Yrs Exp",
                         "current_score": "Score", "risk_level": "Risk"}
            ).sort_values("Score", ascending=False),
            use_container_width=True, hide_index=True,
        )

# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 · FITBIT LIVE FEED
# ───────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📡 Fitbit Live Feed")
    st.caption("Simulated real-time data stream — mirrors TILES-2018 / Fitbit Charge 2 API structure")

    if "ftick" not in st.session_state:
        st.session_state.ftick = 0

    col_btn, col_auto = st.columns([1, 2])
    with col_btn:
        if st.button("🔄 Fetch New Reading"):
            st.session_state.ftick += 1
    with col_auto:
        auto = st.checkbox("Auto-refresh every 4 s (live simulation)")

    tick = st.session_state.ftick
    np.random.seed(tick * 13 + 7)

    cur = {
        "hrv_rmssd":          round(float(np.clip(35 + 10 * np.sin(tick / 3) + np.random.normal(0, 3), 10, 55)), 1),
        "step_regularity":    round(float(np.clip(0.55 + np.random.normal(0, 0.08), 0.1, 0.9)), 2),
        "sleep_efficiency":   round(float(np.clip(82 + np.random.normal(0, 4), 40, 98)), 1),
        "sleep_duration_hrs": round(float(np.clip(6.5 + np.random.normal(0, 0.3), 2, 9)), 1),
        "resting_hr":         round(float(np.clip(70 - 5 * np.sin(tick / 3) + np.random.normal(0, 2), 50, 100)), 1),
        "sedentary_bouts":    max(0, int(np.random.poisson(5))),
        "sleep_onset_var":    round(float(np.clip(0.5 + np.random.normal(0, 0.2), 0.1, 2.0)), 2),
        "n_awakenings":       max(0, int(np.random.poisson(3))),
    }

    st.markdown("#### Current Sensor Readings")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("HRV RMSSD",        f"{cur['hrv_rmssd']} ms",
                delta="Good" if cur["hrv_rmssd"] > 35 else "Low",
                delta_color="normal")
    r1c2.metric("Resting HR",        f"{cur['resting_hr']} bpm",
                delta="Elevated" if cur["resting_hr"] > 75 else "Normal",
                delta_color="inverse")
    r1c3.metric("Sleep Efficiency",  f"{cur['sleep_efficiency']}%",
                delta="Good" if cur["sleep_efficiency"] > 80 else "Poor",
                delta_color="normal")
    r1c4.metric("Sleep Duration",    f"{cur['sleep_duration_hrs']} hrs",
                delta="Sufficient" if cur["sleep_duration_hrs"] > 6 else "Short",
                delta_color="normal")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Step Regularity",   f"{cur['step_regularity']:.2f}",
                delta="Regular" if cur["step_regularity"] > 0.5 else "Irregular",
                delta_color="normal")
    r2c2.metric("Sedentary Bouts",   str(cur["sedentary_bouts"]),
                delta="High" if cur["sedentary_bouts"] > 7 else "OK",
                delta_color="inverse")
    r2c3.metric("Sleep Onset Var.",  f"{cur['sleep_onset_var']} hrs",
                delta="Irregular" if cur["sleep_onset_var"] > 1.0 else "Regular",
                delta_color="inverse")
    r2c4.metric("Night Awakenings",  str(cur["n_awakenings"]),
                delta="Disrupted" if cur["n_awakenings"] > 5 else "Normal",
                delta_color="inverse")

    # Derive burnout estimate from live feed
    live_sv = fitbit_to_signals({
        "hrv_rmssd":              cur["hrv_rmssd"],
        "step_regularity":        cur["step_regularity"],
        "sleep_efficiency":       cur["sleep_efficiency"],
        "sleep_duration_hrs":     cur["sleep_duration_hrs"],
        "sedentary_bouts":        cur["sedentary_bouts"],
        "sleep_onset_variability": cur["sleep_onset_var"],
        "n_awakenings":           cur["n_awakenings"],
    })
    for f in FEATURES:
        live_sv.setdefault(f, SIGNAL_NEUTRAL)

    live_score = composite_score(live_sv)
    live_label = score_to_label(live_score)
    live_proba = clf.predict_proba(np.array([[live_sv[f] for f in FEATURES]]))[0]

    st.divider()
    st.markdown("#### Live Burnout Risk Estimate")
    lc1, lc2 = st.columns([1, 2])
    with lc1:
        risk_badge(live_label)
        st.metric("Live Burnout Score", f"{live_score:.4f}")
    with lc2:
        st.pyplot(prob_chart(live_proba))

    # 24-hour trend (fixed seed for readability)
    st.divider()
    st.markdown("#### 24-Hour Physiological Trend")
    np.random.seed(42)
    hours   = list(range(24))
    hrv_24  = [round(float(35 + 10 * np.sin(i / 3) + np.random.normal(0, 3)), 1) for i in hours]
    hr_24   = [round(float(70 - 5 * np.sin(i / 3) + np.random.normal(0, 2)), 1) for i in hours]
    st.pyplot(hrv_hr_24h_chart(hours, hrv_24, hr_24))

    if auto:
        time.sleep(4)
        st.session_state.ftick += 1
        st.rerun()

# ───────────────────────────────────────────────────────────────────────────────
# TAB 5 · COLOR CHAOS EXERCISE  (Stroop attention test)
# ───────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🎨 Color Chaos Exercise")
    st.markdown(
        "This **Stroop-based attention test** measures cognitive interference resistance — "
        "a validated indicator of burnout-related attention fatigue.  \n"
        "You will see a **color word** displayed in a **mismatched font color**.  \n"
        "Click the **font color** (what you see), not the word's meaning.  \n"
        "10 trials · results map to your _Color Chaos_ burnout signal."
    )

    STROOP = {
        "RED":    "#ef4444",
        "BLUE":   "#3b82f6",
        "GREEN":  "#22c55e",
        "YELLOW": "#eab308",
        "PURPLE": "#a855f7",
    }
    TOTAL = 10

    def _new_trial():
        word  = random.choice(list(STROOP))
        color = random.choice([k for k in STROOP if k != word])
        return word, color

    # Session-state init
    _ss_defaults = {
        "sx_started": False, "sx_finished": False,
        "sx_round": 0, "sx_correct": 0,
        "sx_word": None, "sx_color": None,
        "sx_t0": None, "sx_times": [],
        "sx_last": None,
    }
    for k, v in _ss_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    ss = st.session_state  # shorthand

    if not ss.sx_started and not ss.sx_finished:
        st.info("Press **Start** to begin the 10-trial attention assessment.", icon="🎯")
        if st.button("▶ Start Color Chaos Exercise", type="primary"):
            w, c = _new_trial()
            ss.sx_started, ss.sx_finished = True, False
            ss.sx_round = ss.sx_correct = 0
            ss.sx_times, ss.sx_last = [], None
            ss.sx_word, ss.sx_color = w, c
            ss.sx_t0 = time.time()
            st.rerun()

    elif ss.sx_started and not ss.sx_finished:
        st.progress(ss.sx_round / TOTAL,
                    text=f"Trial {ss.sx_round + 1} of {TOTAL}")

        if ss.sx_last is True:
            st.success("✓ Correct!", icon="✅")
        elif ss.sx_last is False:
            st.error("✗ Incorrect", icon="❌")

        hex_c = STROOP[ss.sx_color]
        st.markdown(
            f"<div style='text-align:center;padding:2.5rem 1rem;margin:1rem 0;"
            f"background:{C['chart_bg']};border-radius:16px;"
            f"border:2px solid {C['border']}'>"
            f"<span style='font-size:5.5rem;font-weight:900;color:{hex_c};"
            f"letter-spacing:.15em'>{ss.sx_word}</span><br>"
            f"<p style='color:{C['muted']};font-size:.9rem;margin-top:.6rem'>"
            f"What COLOR is the text? (ignore the word)</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        btn_cols = st.columns(len(STROOP))
        for i, (cname, chex) in enumerate(STROOP.items()):
            with btn_cols[i]:
                if st.button(cname, key=f"sb_{cname}_{ss.sx_round}",
                             use_container_width=True):
                    elapsed = time.time() - ss.sx_t0
                    correct = (cname == ss.sx_color)
                    ss.sx_times.append(round(elapsed, 2))
                    if correct:
                        ss.sx_correct += 1
                    ss.sx_last  = correct
                    ss.sx_round += 1
                    if ss.sx_round >= TOTAL:
                        ss.sx_finished = True
                        ss.sx_started  = False
                    else:
                        w2, c2 = _new_trial()
                        ss.sx_word, ss.sx_color = w2, c2
                        ss.sx_t0 = time.time()
                    st.rerun()

    elif ss.sx_finished:
        accuracy = ss.sx_correct / TOTAL
        avg_rt   = float(np.mean(ss.sx_times)) if ss.sx_times else 2.0

        # Map performance → chaos signal (higher chaos = worse attention)
        if accuracy >= 0.90 and avg_rt < 3.0:
            interp       = "Excellent cognitive control"
            interp_sub   = "Low cognitive fatigue — attention circuits intact"
            chaos_signal = max(0.05, 0.2 - (accuracy - 0.9) * 2)
            risk_lbl     = "Low"
        elif accuracy >= 0.70:
            interp       = "Moderate attention performance"
            interp_sub   = "Some cognitive interference detected — monitor weekly"
            chaos_signal = 0.45
            risk_lbl     = "Moderate"
        else:
            interp       = "Attention difficulty detected"
            interp_sub   = "High cognitive fatigue — significant burnout indicator"
            chaos_signal = 0.78
            risk_lbl     = "High"

        st.markdown("#### Exercise Results")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Accuracy",           f"{accuracy:.0%}",
                   f"{ss.sx_correct}/{TOTAL} correct")
        rc2.metric("Avg Response Time",  f"{avg_rt:.2f} s", "Lower = sharper")
        rc3.metric("Color Chaos Signal", f"{chaos_signal:.2f}",
                   f"{risk_lbl} fatigue zone")

        rfg = RISK_COLOR[risk_lbl]
        rbg = RISK_BG[risk_lbl]
        rbd = RISK_BORDER[risk_lbl]
        st.markdown(
            f"<div style='background:{rbg};border:2px solid {rbd};"
            f"border-radius:14px;padding:16px 20px;margin:.8rem 0'>"
            f"<strong style='color:{rfg};font-size:1.1rem'>● {interp}</strong><br>"
            f"<span style='color:{C['muted']}'>{interp_sub}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.pyplot(stroop_result_chart(ss.sx_correct, TOTAL, ss.sx_times))

        st.info(
            f"**What this means:** Your Color Chaos signal is estimated at "
            f"**{chaos_signal:.2f}** (scale 0–1). This can be entered manually into "
            f"the Assessment tab under _Color-Pattern Chaos_ to personalise your score.",
            icon="ℹ️",
        )

        if st.button("🔄 Retake Exercise"):
            for k in list(_ss_defaults.keys()):
                del st.session_state[k]
            st.rerun()
