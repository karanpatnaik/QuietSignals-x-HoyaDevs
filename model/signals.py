#Palette
BG    = "#f0fdf4"
PANEL = "#ffffff"
GREEN = "#22c55e"
AMBER = "#f59e0b"
RED   = "#ef4444"
BLUE  = "#4ade80"
TEXT  = "#14532d"
MUTED = "#6b7280"

CLASS_COLORS = [GREEN, AMBER, RED]
CLASS_NAMES  = ["Low", "Moderate", "High"]

# Signal definitions
# Weights sum to 1.0. Signals are grouped into three sources:
#   - Behavioral/wearable (gsr, task_switch, voice_monotony, gait_irregularity,
#     patient_rel, color_chaos, tiktok_burnout)  → total weight 0.80
#   - Facial emotion via DeepFace (facial_negative_load, facial_flat_affect,
#     facial_positive_protect)                   → total weight 0.20
SIGNALS = {
    # --- Behavioral / wearable signals ---
    "gsr":                     {"label": "Galvanic Skin Response",        "weight": 0.16, "inverse": False},
    "task_switch":             {"label": "Cognitive Task-Switch Burden",   "weight": 0.14, "inverse": False},
    "voice_monotony":          {"label": "Voice Monotony Index",           "weight": 0.13, "inverse": False},
    "gait_irregularity":       {"label": "Gait Irregularity",              "weight": 0.10, "inverse": False},
    "patient_rel":             {"label": "Patient Relationship Quality",   "weight": 0.10, "inverse": True},
    "color_chaos":             {"label": "Stroke Irregularity / Color Negativity", "weight": 0.09, "inverse": False},
    "tiktok_burnout":          {"label": "TikTok Burnout Signal",          "weight": 0.08, "inverse": False},
    # --- Facial emotion signals (from grayscale.py / DeepFace) ---
    "facial_negative_load":    {"label": "Facial Negative Emotion Load",   "weight": 0.10, "inverse": False},
    "facial_flat_affect":      {"label": "Facial Flat Affect Index",       "weight": 0.06, "inverse": False},
    "facial_positive_protect": {"label": "Facial Positive Affect",         "weight": 0.04, "inverse": True},
}

FEATURES = list(SIGNALS.keys())

# Strategy definitions for synthetic data generation
STRATEGIES = {
    "gsr":                     {"method": "real_calibrated", "stressed_mean": 0.72, "neutral_mean": 0.28, "std": 0.09},
    "task_switch":             {"method": "synthetic", "slope": 0.75, "base": 0.10},
    "voice_monotony":          {"method": "real_calibrated", "stressed_mean": 0.68, "neutral_mean": 0.25, "std": 0.10},
    "gait_irregularity":       {"method": "real_calibrated", "stressed_mean": 0.62, "neutral_mean": 0.22, "std": 0.08},
    "patient_rel":             {"method": "synthetic", "slope": 0.75, "base": 0.15, "inverse": True},
    "color_chaos":             {"method": "synthetic", "slope": 0.58, "base": 0.10},
    "tiktok_burnout":          {"method": "synthetic", "slope": 0.50, "base": 0.08},
    "facial_negative_load":    {"method": "synthetic", "slope": 0.55, "base": 0.05},
    "facial_flat_affect":      {"method": "synthetic", "slope": 0.45, "base": 0.15},
    "facial_positive_protect": {"method": "synthetic", "slope": 0.70, "base": 0.10, "inverse": True},
}
