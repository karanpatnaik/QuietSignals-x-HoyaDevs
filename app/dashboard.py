import sys, os, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train import train_model
from model.predict import composite_score, SIGNAL_NEUTRAL
from model.signals import SIGNALS, FEATURES, CLASS_NAMES
from model.fitbit import fitbit_to_signals

try:
    from model.grayscale import grayscale_to_signals
    GRAYSCALE_AVAILABLE = True
except ImportError:
    GRAYSCALE_AVAILABLE = False
    def grayscale_to_signals(report):
        clip = lambda v: round(max(0.0, min(1.0, v)), 4)
        return {
            "facial_negative_load":    clip(report.get("avg_negative", 0) / 60),
            "facial_flat_affect":      clip(report.get("avg_neutral",  0) / 90),
            "facial_positive_protect": clip(report.get("avg_happy",    0) / 60),
        }

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="QuietSignals", layout="wide", page_icon="Q")

# ── palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":       "#080c14",
    "surface":  "#0d1117",
    "panel":    "#111827",
    "panel2":   "#161f2e",
    "border":   "#1e2d3d",
    "teal":     "#00c9a7",
    "teal_dim": "#00876f",
    "amber":    "#f5a623",
    "red":      "#e8384f",
    "blue":     "#4a9eff",
    "text":     "#dde3ed",
    "muted":    "#57677a",
    "low_fg":   "#00c9a7",  "low_bg":  "#071f1a",  "low_bd":  "#00524a",
    "mod_fg":   "#f5a623",  "mod_bg":  "#1f1507",  "mod_bd":  "#6b4810",
    "hi_fg":    "#e8384f",  "hi_bg":   "#1f0a0d",  "hi_bd":   "#6b1520",
}

RISK_COLOR  = {"Low": C["low_fg"], "Moderate": C["mod_fg"], "High": C["hi_fg"]}
RISK_BG     = {"Low": C["low_bg"], "Moderate": C["mod_bg"], "High": C["hi_bg"]}
RISK_BORDER = {"Low": C["low_bd"], "Moderate": C["mod_bd"], "High": C["hi_bd"]}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Sora:wght@300;400;500;600;700&display=swap');

  html, body, .main, [data-testid="stAppViewContainer"] {{
    background: {C['bg']} !important;
    font-family: 'Sora', sans-serif;
    color: {C['text']};
  }}
  [data-testid="stSidebar"] {{
    background: {C['surface']} !important;
    border-right: 1px solid {C['border']};
  }}
  [data-testid="stSidebar"] * {{ color: {C['text']} !important; }}
  h1, h2, h3, h4 {{
    font-family: 'Sora', sans-serif !important;
    color: {C['text']} !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
  }}
  p, label, span {{ color: {C['text']} !important; }}
  [data-testid="metric-container"] {{
    background: {C['panel']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
  }}
  [data-testid="stMetricLabel"] {{
    color: {C['muted']} !important;
    font-size: .72rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: .08em;
  }}
  [data-testid="stMetricValue"] {{
    color: {C['teal']} !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
  }}
  [data-testid="stMetricDelta"] svg {{ display: none; }}
  .stTabs [data-baseweb="tab-list"] {{
    background: {C['surface']};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {C['border']};
    gap: 2px;
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 7px;
    padding: 7px 22px;
    color: {C['muted']};
    font-family: 'Sora', sans-serif;
    font-size: .85rem;
  }}
  .stTabs [aria-selected="true"] {{
    background: {C['teal']} !important;
    color: #000 !important;
    font-weight: 600;
  }}
  .stButton > button {{
    background: {C['teal']};
    color: #000;
    border: none;
    border-radius: 7px;
    font-weight: 600;
    padding: .45rem 1.4rem;
    font-family: 'Sora', sans-serif;
    font-size: .85rem;
    letter-spacing: .01em;
    transition: opacity .15s;
  }}
  .stButton > button:hover {{ opacity: .82; color: #000; }}
  .stButton > button[kind="secondary"] {{
    background: transparent;
    color: {C['teal']};
    border: 1px solid {C['teal_dim']};
  }}
  [data-testid="stSelectbox"] > div > div,
  [data-testid="stTextInput"] > div > div > input {{
    background: {C['panel']} !important;
    border-color: {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 7px !important;
    font-family: 'Sora', sans-serif !important;
  }}
  .stSlider [data-testid="stMarkdownContainer"] p {{
    color: {C['muted']} !important;
    font-size: .8rem !important;
  }}
  [data-testid="stExpander"] {{
    background: {C['panel']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 10px !important;
  }}
  details summary {{
    color: {C['text']} !important;
    font-weight: 500 !important;
    font-size: .88rem !important;
  }}
  .stDataFrame {{ background: {C['panel']}; border-radius: 10px; }}
  .stDataFrame [data-testid="stDataFrameResizable"] {{ border-radius: 10px; }}
  hr {{ border-color: {C['border']} !important; margin: 1.2rem 0 !important; }}
  .stProgress > div > div {{ background: {C['teal']} !important; }}
  .stAlert {{ border-radius: 10px !important; }}
  .stRadio [data-testid="stMarkdownContainer"] p {{
    font-size: .82rem !important;
    color: {C['muted']} !important;
  }}
  .label-mono {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
    color: {C['muted']};
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: .3rem;
  }}
  div[data-testid="stNumberInput"] input {{
    background: {C['panel']} !important;
    border-color: {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 7px !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return train_model()

clf = get_model()

# ── helpers ───────────────────────────────────────────────────────────────────
def score_to_label(s):
    return "Low" if s < 0.33 else ("Moderate" if s < 0.55 else "High")

def section_label(text):
    st.markdown(f"<div class='label-mono'>{text}</div>", unsafe_allow_html=True)

def risk_badge(label, score=None, large=False):
    size = "1.6rem" if large else "1rem"
    pad  = "12px 28px" if large else "6px 18px"
    score_str = (f"&nbsp;&nbsp;<span style='font-size:.85rem;opacity:.7;"
                 f"font-family:IBM Plex Mono,monospace'>{score:.4f}</span>") if score is not None else ""
    st.markdown(
        f"<div style='background:{RISK_BG[label]};color:{RISK_COLOR[label]};"
        f"padding:{pad};border:1.5px solid {RISK_BORDER[label]};border-radius:40px;"
        f"font-size:{size};font-weight:600;display:inline-block;"
        f"font-family:Sora,sans-serif;letter-spacing:.01em'>"
        f"{label} Risk{score_str}</div>",
        unsafe_allow_html=True,
    )

def info_card(html, bg=None, border=None):
    st.markdown(
        f"<div style='background:{bg or C['panel2']};border:1px solid {border or C['border']};"
        f"border-radius:10px;padding:14px 18px;margin:5px 0;line-height:1.55'>{html}</div>",
        unsafe_allow_html=True,
    )

# ── nurse database ────────────────────────────────────────────────────────────
if "nurse_db"   not in st.session_state: st.session_state.nurse_db   = {}
if "history_db" not in st.session_state: st.session_state.history_db = {}

def get_nurse(nid):       return st.session_state.nurse_db.get(nid)
def save_nurse(nid, rec): st.session_state.nurse_db[nid] = rec

def push_history(nid, score, label, signals):
    h = st.session_state.history_db.setdefault(nid, [])
    h.append({"week": len(h) + 1, "score": round(score, 4),
               "label": label, "signals": signals.copy(),
               "ts": time.strftime("%Y-%m-%d")})

def warning_flags(sv):
    flags = []
    high_t = {
        "gsr":                  ("High stress arousal",               0.65),
        "task_switch":          ("Cognitive overload",                 0.65),
        "voice_monotony":       ("Vocal disengagement",               0.60),
        "gait_irregularity":    ("Movement irregularity",             0.60),
        "color_chaos":          ("Elevated stroke / color negativity", 0.65),
        "tiktok_burnout":       ("Sleep disruption",                  0.65),
        "facial_negative_load": ("Negative emotional load",           0.55),
        "facial_flat_affect":   ("Flat affect / depersonalization",   0.55),
    }
    low_t = {"patient_rel": ("Poor patient relationship quality", 0.35)}
    for sig, (lbl, th) in high_t.items():
        if sv.get(sig, SIGNAL_NEUTRAL) >= th:
            flags.append(("HIGH", lbl, sig, sv.get(sig, SIGNAL_NEUTRAL)))
    for sig, (lbl, th) in low_t.items():
        if sv.get(sig, SIGNAL_NEUTRAL) <= th:
            flags.append(("LOW", lbl, sig, sv.get(sig, SIGNAL_NEUTRAL)))
    return flags

def ai_suggestions(sv, label):
    tips = []
    if sv.get("gsr", .5)               > .65: tips.append("Practice 4-7-8 breathing between patient interactions to reduce sympathetic activation.")
    if sv.get("task_switch", .5)        > .65: tips.append("Discuss task batching with charge nurse — grouped documentation reduces cognitive interrupt load.")
    if sv.get("voice_monotony", .5)     > .60: tips.append("Schedule brief daily peer check-ins. Social engagement reduces emotional disengagement.")
    if sv.get("gait_irregularity", .5)  > .60: tips.append("Check hydration and footwear fit — irregular gait may indicate fatigue or musculoskeletal strain.")
    if sv.get("patient_rel", .5)        < .35: tips.append("Patient relationship quality declining — consider EAP referral for compassion fatigue support.")
    if sv.get("color_chaos", .5)        > .65: tips.append("Elevated stroke irregularity detected. Short break and hydration check recommended.")
    if sv.get("tiktok_burnout", .5)     > .65: tips.append("High sleep disruption signal. Reduce screen time one hour before sleep; consider a sleep diary.")
    if sv.get("facial_negative_load",.5)> .55: tips.append("Elevated negative emotional expression — mental health check-in or EAP consultation recommended.")
    if sv.get("facial_flat_affect", .5) > .55: tips.append("Reduced expressivity may indicate depersonalization. Peer support or engagement program advised.")
    if label == "High":
        tips.insert(0, "Notify charge nurse for an immediate well-being check-in.")
    if not tips:
        tips.append("All signals within healthy range. Maintain current self-care routines and check in monthly.")
    return tips[:6]

# ── plotly theme ──────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=C["panel"],
    font=dict(family="Sora", color=C["text"], size=11),
    margin=dict(l=44, r=20, t=44, b=36),
)

def pl_axes(fig, x_grid=False):
    fig.update_yaxes(gridcolor=C["border"], zeroline=False)
    fig.update_xaxes(gridcolor=C["border"] if x_grid else "rgba(0,0,0,0)", zeroline=False)
    return fig

# ── charts ────────────────────────────────────────────────────────────────────
def chart_prob(proba):
    fig = go.Figure(go.Bar(
        x=CLASS_NAMES, y=proba,
        marker_color=[C["low_fg"], C["amber"], C["hi_fg"]],
        text=[f"{p:.1%}" for p in proba],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11),
        hovertemplate="<b>%{x}</b><br>%{y:.1%}<extra></extra>",
        width=0.45,
    ))
    fig.update_layout(**PL,
        title=dict(text="Class Probability", font=dict(size=12), x=0, y=.97),
        yaxis=dict(tickformat=".0%", range=[0, 1.18], gridcolor=C["border"]),
        height=260, showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig

def chart_radar(sv):
    short = {
        "gsr": "GSR", "task_switch": "Task Switch",
        "voice_monotony": "Voice", "gait_irregularity": "Gait",
        "patient_rel": "Patient Rel.", "color_chaos": "Color Chaos",
        "tiktok_burnout": "TikTok", "facial_negative_load": "Neg. Load",
        "facial_flat_affect": "Flat Affect", "facial_positive_protect": "Pos. Affect",
    }
    labels = [short[f] for f in FEATURES]
    vals   = [sv.get(f, SIGNAL_NEUTRAL) for f in FEATURES]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(0,201,167,0.12)",
        line=dict(color=C["teal"], width=2),
        marker=dict(size=5, color=C["teal"]),
        hovertemplate="<b>%{theta}</b><br>%{r:.3f}<extra></extra>",
    ))
    fig.update_layout(**PL,
        polar=dict(
            bgcolor=C["panel"],
            radialaxis=dict(range=[0, 1], gridcolor=C["border"], tickfont=dict(size=8, color=C["muted"]),
                            tickvals=[0.25, 0.5, 0.75], tickcolor="rgba(0,0,0,0)"),
            angularaxis=dict(gridcolor=C["border"], tickfont=dict(size=9, color=C["text"])),
        ),
        title=dict(text="Signal Radar", font=dict(size=12), x=0, y=.97),
        height=300, showlegend=False,
    )
    return fig

def chart_contribution(sv):
    labels, values, colors = [], [], []
    for f in FEATURES:
        v = sv.get(f, SIGNAL_NEUTRAL)
        c = (1 - v if SIGNALS[f]["inverse"] else v) * SIGNALS[f]["weight"]
        labels.append(SIGNALS[f]["label"])
        values.append(c)
        colors.append(C["hi_fg"] if c > 0.08 else (C["amber"] if c > 0.04 else C["teal"]))
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10),
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
        width=0.55,
    ))
    fig.update_layout(**PL,
        title=dict(text="Signal Contributions to Burnout Score", font=dict(size=12), x=0, y=.97),
        xaxis=dict(gridcolor=C["border"]),
        yaxis=dict(autorange="reversed"),
        height=360, showlegend=False,
    )
    fig.update_xaxes(gridcolor=C["border"])
    return fig

def chart_trend(weeks, scores, forecast=None, name=""):
    fig = go.Figure()
    fig.add_hrect(y0=0,    y1=0.33, fillcolor=C["teal"],  opacity=0.035, line_width=0)
    fig.add_hrect(y0=0.33, y1=0.55, fillcolor=C["amber"], opacity=0.035, line_width=0)
    fig.add_hrect(y0=0.55, y1=1.0,  fillcolor=C["hi_fg"], opacity=0.035, line_width=0)
    for y, lbl, col in [(0.33, "Moderate", C["amber"]), (0.55, "High", C["hi_fg"])]:
        fig.add_hline(y=y, line_dash="dot", line_color=col, opacity=0.45,
                      annotation_text=lbl,
                      annotation_font=dict(color=col, size=9),
                      annotation_position="top right")
    fig.add_trace(go.Scatter(
        x=weeks, y=scores, mode="lines+markers", name="Score",
        line=dict(color=C["teal"], width=2.5),
        marker=dict(size=7, color=C["teal"], line=dict(width=1.5, color=C["bg"])),
        fill="tozeroy", fillcolor="rgba(0,201,167,0.07)",
        hovertemplate="Week %{x}<br>Score: %{y:.4f}<extra></extra>",
    ))
    if forecast:
        fw = [weeks[-1] + i for i in range(1, len(forecast) + 1)]
        fig.add_trace(go.Scatter(
            x=[weeks[-1]] + fw, y=[scores[-1]] + forecast,
            mode="lines+markers", name="Forecast",
            line=dict(color=C["blue"], width=2, dash="dash"),
            marker=dict(size=6, color=C["blue"], line=dict(width=1.5, color=C["bg"])),
            hovertemplate="Week %{x} (forecast)<br>%{y:.4f}<extra></extra>",
        ))
    fig.update_layout(**PL,
        title=dict(text=f"Burnout Score Trend{(' — ' + name) if name else ''}",
                   font=dict(size=12), x=0, y=.97),
        yaxis=dict(range=[0, 1], gridcolor=C["border"], title="Score"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", title="Week"),
        height=300,
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def chart_dept_bar(dept_stats):
    colors = [C["hi_fg"] if s >= 0.55 else (C["amber"] if s >= 0.33 else C["teal"])
              for s in dept_stats["avg_score"]]
    fig = go.Figure(go.Bar(
        x=dept_stats["department"], y=dept_stats["avg_score"],
        marker_color=colors,
        text=[f"{s:.3f}" for s in dept_stats["avg_score"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11),
        hovertemplate="<b>%{x}</b><br>Avg Score: %{y:.3f}<extra></extra>",
        width=0.5,
    ))
    for y, lbl, col in [(0.33, "Moderate", C["amber"]), (0.55, "High", C["hi_fg"])]:
        fig.add_hline(y=y, line_dash="dot", line_color=col, opacity=0.5,
                      annotation_text=lbl,
                      annotation_font=dict(color=col, size=9),
                      annotation_position="top right")
    fig.update_layout(**PL,
        title=dict(text="Average Burnout Score by Department", font=dict(size=12), x=0, y=.97),
        yaxis=dict(range=[0, 1], gridcolor=C["border"]),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=320, showlegend=False,
    )
    return fig

def chart_tap_intervals(tap_times):
    intervals = [tap_times[i+1] - tap_times[i] for i in range(len(tap_times)-1)]
    avg = float(np.mean(intervals))
    fig = go.Figure()
    fig.add_hline(y=avg, line_dash="dot", line_color=C["amber"],
                  annotation_text=f"Mean {avg:.2f}s",
                  annotation_font=dict(color=C["amber"], size=9))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(intervals)+1)), y=intervals,
        mode="lines+markers",
        line=dict(color=C["teal"], width=2),
        marker=dict(size=7, color=C["teal"], line=dict(width=1.5, color=C["bg"])),
        fill="tozeroy", fillcolor="rgba(0,201,167,0.07)",
        hovertemplate="Interval %{x}<br>%{y:.3f}s<extra></extra>",
        name="Tap interval",
    ))
    fig.update_layout(**PL,
        title=dict(text="Tap Interval Regularity", font=dict(size=12), x=0, y=.97),
        yaxis=dict(title="Interval (s)", gridcolor=C["border"]),
        xaxis=dict(title="Interval #", gridcolor="rgba(0,0,0,0)"),
        height=220, showlegend=False,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='padding:.8rem 0 .6rem 0'>
      <div style='font-size:1.1rem;font-weight:700;color:{C["teal"]};
                  font-family:IBM Plex Mono,monospace;letter-spacing:-.01em'>
        QuietSignals
      </div>
      <div style='font-size:.72rem;color:{C["muted"]};margin-top:2px'>
        Burnout Intelligence &nbsp;·&nbsp; MedStar Georgetown
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("", ["Start Analysis", "Personal History", "Hospital Overview"],
                    label_visibility="collapsed")
    st.divider()

    section_label("Nurse Lookup")
    lookup_id = st.text_input("", placeholder="Nurse ID — e.g. RN-1042",
                              label_visibility="collapsed", key="sidebar_lookup")
    if lookup_id:
        found = get_nurse(lookup_id.strip().upper())
        if found:
            st.markdown(
                f"<div style='background:{C['panel']};border:1px solid {C['teal_dim']};"
                f"border-radius:8px;padding:10px 14px;margin-top:4px'>"
                f"<div style='color:{C['teal']};font-weight:600;font-size:.88rem'>{found['name']}</div>"
                f"<div style='color:{C['muted']};font-size:.76rem;margin-top:2px'>"
                f"{found['id']} &nbsp;·&nbsp; {found['dept']} &nbsp;·&nbsp; {found['shift']}"
                f"</div></div>", unsafe_allow_html=True,
            )
        else:
            st.markdown(f"<div style='color:{C['muted']};font-size:.78rem;margin-top:4px'>No record found.</div>",
                        unsafe_allow_html=True)

    st.divider()
    nurse_count = len(st.session_state.nurse_db)
    st.markdown(
        f"<div style='font-size:.75rem;color:{C['muted']};font-family:IBM Plex Mono,monospace'>"
        f"{nurse_count} nurse{'s' if nurse_count != 1 else ''} registered</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — START ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Start Analysis":
    st.markdown("<h1 style='margin-bottom:.15rem'>Start Analysis</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{C['muted']};font-size:.85rem;margin-bottom:1.2rem'>"
                f"Run a burnout assessment. Save results to the nurse's record for longitudinal tracking.</div>",
                unsafe_allow_html=True)
    st.divider()

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        section_label("Nurse Identification")
        ca, cb = st.columns(2)
        with ca: n_name  = st.text_input("Full Name",  placeholder="e.g. Sarah Chen")
        with cb: n_id    = st.text_input("Nurse ID",   placeholder="e.g. RN-1042").strip().upper()
        cc, cd = st.columns(2)
        with cc: n_dept  = st.selectbox("Department",  ["—","ICU","ER","Med-Surg","Oncology","Pediatrics","NICU","OR","Other"])
        with cd: n_shift = st.selectbox("Shift",       ["—","Day","Night","Rotating"])
        n_yrs = st.number_input("Years of Experience", min_value=0, max_value=45, value=3)
        identified = bool(n_name and n_id and n_dept != "—")

        st.divider()
        section_label("Signal Source")
        input_mode = st.radio("", ["Manual Input", "Fitbit Sensors", "Manual + Facial"],
                              label_visibility="collapsed")

        st.divider()
        section_label("Signal Values")
        sv: dict = {}

        if input_mode == "Manual Input":
            for f in [x for x in FEATURES if not x.startswith("facial")]:
                sv[f] = st.slider(SIGNALS[f]["label"], 0.0, 1.0, 0.5, 0.01, key=f"m_{f}")
            st.caption("Facial signals set to neutral (0.5).")

        elif input_mode == "Fitbit Sensors":
            hrv       = st.slider("HRV RMSSD (ms)",           10.0, 55.0, 35.0, 0.5)
            step_reg  = st.slider("Step Regularity",            0.1,  0.8,  0.5, 0.01)
            sleep_eff = st.slider("Sleep Efficiency (%)",      40.0, 98.0, 80.0, 0.5)
            sleep_hrs = st.slider("Sleep Duration (hrs)",       2.0,  9.0,  6.5, 0.1)
            sed_bouts = st.slider("Sedentary Bouts",              0,   14,    5)
            sov       = st.slider("Sleep Onset Variability",   0.1,  2.0,  0.6, 0.05)
            n_awk     = st.slider("Night Awakenings",             0,   12,    3)
            sv.update(fitbit_to_signals({
                "hrv_rmssd": hrv, "step_regularity": step_reg,
                "sleep_efficiency": sleep_eff, "sleep_duration_hrs": sleep_hrs,
                "sedentary_bouts": sed_bouts, "sleep_onset_variability": sov,
                "n_awakenings": n_awk,
            }))
            st.caption("Facial signals set to neutral (0.5).")

        else:
            for f in [x for x in FEATURES if not x.startswith("facial")]:
                sv[f] = st.slider(SIGNALS[f]["label"], 0.0, 1.0, 0.5, 0.01, key=f"mf_{f}")
            st.divider()
            section_label("Facial Signals")
            avg_neg = st.slider("Avg Negative Emotion (%)", 0.0, 100.0, 20.0, 1.0)
            avg_neu = st.slider("Avg Neutral Expression (%)", 0.0, 100.0, 50.0, 1.0)
            avg_hap = st.slider("Avg Happy Expression (%)", 0.0, 100.0, 20.0, 1.0)
            sv.update(grayscale_to_signals({"avg_negative": avg_neg,
                                            "avg_neutral": avg_neu, "avg_happy": avg_hap}))

        for f in FEATURES:
            sv.setdefault(f, SIGNAL_NEUTRAL)

    # ── results ───────────────────────────────────────────────────────────────
    with right:
        score = composite_score(sv)
        label = score_to_label(score)
        proba = clf.predict_proba(np.array([[sv[f] for f in FEATURES]]))[0]

        section_label("Assessment Result")
        risk_badge(label, score, large=True)
        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Composite Score",  f"{score:.4f}")
        m2.metric("ML Confidence",    f"{max(proba):.0%}")
        m3.metric("Signals Analyzed", f"{len(FEATURES)}")

        st.markdown("<br>", unsafe_allow_html=True)

        # signal breakdown
        with st.expander("Signal Breakdown", expanded=True):
            rows = []
            for f in FEATURES:
                v = sv[f]
                contrib = (1 - v if SIGNALS[f]["inverse"] else v) * SIGNALS[f]["weight"]
                rows.append({
                    "Signal":        SIGNALS[f]["label"],
                    "Value":         f"{v:.3f}",
                    "Contribution":  f"{contrib:.4f}",
                    "Status":        "High" if contrib > 0.08 else ("Moderate" if contrib > 0.04 else "Low"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # warning flags
        flags = warning_flags(sv)
        if flags:
            with st.expander(f"Warning Flags — {len(flags)} signal(s) elevated", expanded=True):
                for marker, lbl, _sig, val in flags:
                    info_card(
                        f"<span style='color:{C['hi_fg']};font-family:IBM Plex Mono,monospace;"
                        f"font-size:.8rem'>[{marker}]</span>"
                        f"&nbsp;&nbsp;<span style='color:{C['text']}'>{lbl}</span>"
                        f"&nbsp;&nbsp;<span style='color:{C['muted']};font-size:.82rem'>{val:.3f}</span>",
                        bg=C["hi_bg"], border=C["hi_bd"],
                    )
        else:
            st.success("All signals within healthy range.")

        # clinical suggestions
        tips = ai_suggestions(sv, label)
        with st.expander("Clinical Suggestions", expanded=(label != "Low")):
            for i, tip in enumerate(tips):
                priority = (label == "High" and i == 0)
                info_card(
                    f"<span style='color:{C['hi_fg'] if priority else C['teal']}'>{tip}</span>",
                    bg=C["hi_bg"] if priority else C["panel2"],
                    border=C["hi_bd"] if priority else C["border"],
                )

        # charts
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_prob(proba), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            st.plotly_chart(chart_radar(sv), use_container_width=True,
                            config={"displayModeBar": False})

        with st.expander("Signal Contributions"):
            st.plotly_chart(chart_contribution(sv), use_container_width=True,
                            config={"displayModeBar": False})

        # save
        st.divider()
        if identified:
            if st.button("Save to Nurse Record", use_container_width=True):
                if not get_nurse(n_id):
                    save_nurse(n_id, {"id": n_id, "name": n_name, "dept": n_dept,
                                      "shift": n_shift, "years_exp": n_yrs})
                push_history(n_id, score, label, sv)
                st.success(f"Assessment saved for {n_name} ({n_id}).")
        else:
            st.info("Complete nurse identification to enable saving.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PERSONAL HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Personal History":
    st.markdown("<h1 style='margin-bottom:.15rem'>Personal History</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{C['muted']};font-size:.85rem;margin-bottom:1.2rem'>"
                f"Longitudinal burnout tracking — select a nurse to view their full timeline.</div>",
                unsafe_allow_html=True)
    st.divider()

    db = st.session_state.nurse_db
    if not db:
        st.info("No nurses registered. Run an assessment and save it to begin tracking.")
        st.stop()

    nurse_options = {f"{v['name']} ({k})": k for k, v in db.items()}
    sel_label = st.selectbox("", list(nurse_options.keys()), label_visibility="collapsed")
    sel_id    = nurse_options[sel_label]
    nurse     = db[sel_id]
    history   = st.session_state.history_db.get(sel_id, [])

    # profile card
    ls = history[-1]["score"] if history else 0.0
    ll = history[-1]["label"] if history else "Low"
    st.markdown(
        f"<div style='background:{RISK_BG[ll]};border:1.5px solid {RISK_BORDER[ll]};"
        f"border-radius:12px;padding:18px 24px;margin:1rem 0;"
        f"display:flex;justify-content:space-between;align-items:center'>"
        f"<div>"
        f"<div style='color:{RISK_COLOR[ll]};font-size:1.2rem;font-weight:700'>{nurse['name']}</div>"
        f"<div style='color:{C['muted']};font-size:.8rem;margin-top:4px'>"
        f"{nurse['id']} &nbsp;·&nbsp; {nurse['dept']} &nbsp;·&nbsp; {nurse['shift']} Shift"
        f" &nbsp;·&nbsp; {nurse['years_exp']} yrs experience</div>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<div style='color:{RISK_COLOR[ll]};font-size:1.6rem;font-weight:700;"
        f"font-family:IBM Plex Mono,monospace'>{ls:.4f}</div>"
        f"<div style='color:{RISK_COLOR[ll]};font-size:.8rem'>{ll} Risk</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if not history:
        st.info("No assessments recorded for this nurse yet.")
        st.stop()

    scores = [h["score"] for h in history]
    weeks  = [h["week"]  for h in history]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Score",   f"{scores[-1]:.4f}")
    m2.metric("Average Score",  f"{np.mean(scores):.4f}")
    m3.metric("Peak Score",     f"{max(scores):.4f}")
    m4.metric("Assessments",    str(len(history)))

    st.markdown("<br>", unsafe_allow_html=True)

    # trend + stats
    coeffs = np.polyfit(weeks, scores, 1) if len(weeks) > 1 else [0, scores[0]]
    fw = [weeks[-1] + i for i in range(1, 5)]
    np.random.seed(abs(hash(sel_id)) % (2**31))
    forecast = [round(float(np.clip(np.polyval(coeffs, w) + np.random.normal(0, 0.025), 0.05, 0.95)), 4)
                for w in fw]
    trend_dir = "Worsening" if coeffs[0] > 0.01 else ("Improving" if coeffs[0] < -0.01 else "Stable")

    tc1, tc2 = st.columns([2.2, 1])
    with tc1:
        st.plotly_chart(chart_trend(weeks, scores, forecast, nurse["name"]),
                        use_container_width=True, config={"displayModeBar": False})
    with tc2:
        st.markdown("<br>", unsafe_allow_html=True)
        section_label("Trend Stats")
        st.metric("Direction",       trend_dir)
        st.metric("Change / Week",   f"{coeffs[0]:+.4f}")
        st.metric("4-Week Forecast", f"{forecast[-1]:.4f}",
                  delta=score_to_label(forecast[-1]))

    st.divider()

    # per-session history
    section_label("Assessment History")
    st.markdown("<br>", unsafe_allow_html=True)
    for h in reversed(history):
        with st.expander(
            f"Week {h['week']}  ·  {h['ts']}  ·  Score {h['score']:.4f}  ·  {h['label']} Risk"
        ):
            risk_badge(h["label"])
            st.markdown("<br>", unsafe_allow_html=True)
            rows = []
            for f in FEATURES:
                v = h["signals"].get(f, SIGNAL_NEUTRAL)
                contrib = (1 - v if SIGNALS[f]["inverse"] else v) * SIGNALS[f]["weight"]
                rows.append({
                    "Signal":       SIGNALS[f]["label"],
                    "Value":        f"{v:.3f}",
                    "Contribution": f"{contrib:.4f}",
                    "Status":       "High" if contrib > 0.08 else ("Moderate" if contrib > 0.04 else "Low"),
                })
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with c2:
                st.plotly_chart(chart_radar(h["signals"]), use_container_width=True,
                                config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HOSPITAL OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Hospital Overview":
    st.markdown("<h1 style='margin-bottom:.15rem'>Hospital Overview</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{C['muted']};font-size:.85rem;margin-bottom:1.2rem'>"
                f"Aggregate burnout analytics across all registered nurses and departments.</div>",
                unsafe_allow_html=True)
    st.divider()

    db      = st.session_state.nurse_db
    hist_db = st.session_state.history_db

    if not db:
        st.info("No nurses registered. Add nurses via Start Analysis to populate this view.")
        st.stop()

    rows = []
    for nid, nurse in db.items():
        h = hist_db.get(nid, [])
        if h:
            latest = h[-1]
            rows.append({
                "id": nid, "name": nurse["name"], "department": nurse["dept"],
                "shift": nurse["shift"], "years_exp": nurse["years_exp"],
                "current_score": latest["score"], "risk_level": latest["label"],
                "n_assessments": len(h),
            })

    if not rows:
        st.info("Nurses are registered but have no assessments yet.")
        st.stop()

    profiles = pd.DataFrame(rows)
    total  = len(profiles)
    n_high = (profiles["risk_level"] == "High").sum()
    n_mod  = (profiles["risk_level"] == "Moderate").sum()
    n_low  = (profiles["risk_level"] == "Low").sum()
    avg_s  = profiles["current_score"].mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Nurses",        total)
    m2.metric("Hospital Avg Score",  f"{avg_s:.3f}")
    m3.metric("High Risk",           n_high, f"{n_high/total:.0%} of staff")
    m4.metric("Moderate Risk",       n_mod,  f"{n_mod/total:.0%} of staff")
    m5.metric("Low Risk",            n_low,  f"{n_low/total:.0%} of staff")

    st.markdown("<br>", unsafe_allow_html=True)

    if profiles["department"].nunique() > 1:
        dept_stats = (
            profiles.groupby("department")
            .agg(avg_score=("current_score","mean"), n_nurses=("id","count"),
                 n_high=("risk_level", lambda x: (x=="High").sum()),
                 n_moderate=("risk_level", lambda x: (x=="Moderate").sum()),
                 n_low=("risk_level", lambda x: (x=="Low").sum()))
            .reset_index()
        )
        dept_stats["avg_score"] = dept_stats["avg_score"].round(3)

        ch_col, tbl_col = st.columns([1.5, 1])
        with ch_col:
            st.plotly_chart(chart_dept_bar(dept_stats), use_container_width=True,
                            config={"displayModeBar": False})
        with tbl_col:
            section_label("Department Summary")
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(
                dept_stats.rename(columns={"department":"Dept","n_nurses":"Nurses",
                                           "avg_score":"Avg Score","n_high":"High",
                                           "n_moderate":"Moderate","n_low":"Low"}),
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # flagged nurses
    section_label("Requires Attention")
    st.markdown("<br>", unsafe_allow_html=True)
    flagged = profiles[profiles["risk_level"].isin(["High","Moderate"])].sort_values("current_score", ascending=False)
    if flagged.empty:
        st.success("No nurses currently at Moderate or High risk.")
    else:
        st.dataframe(
            flagged[["id","name","department","shift","current_score","risk_level","n_assessments"]].rename(
                columns={"id":"ID","name":"Name","department":"Dept","shift":"Shift",
                         "current_score":"Score","risk_level":"Risk","n_assessments":"Assessments"}
            ),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    with st.expander(f"All Nurses ({total})"):
        st.dataframe(
            profiles[["id","name","department","shift","years_exp","current_score","risk_level","n_assessments"]].rename(
                columns={"id":"ID","name":"Name","department":"Dept","shift":"Shift",
                         "years_exp":"Yrs Exp","current_score":"Score",
                         "risk_level":"Risk","n_assessments":"Assessments"}
            ).sort_values("Score", ascending=False),
            use_container_width=True, hide_index=True,
        )
