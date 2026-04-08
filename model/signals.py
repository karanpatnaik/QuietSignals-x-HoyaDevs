#Palette
BG    = "#0a0d14"
PANEL = "#111520"
TEAL  = "#00d4b8"
AMBER = "#f5a623"
RED   = "#ff4e6a"
BLUE  = "#4a9eff"
TEXT  = "#e2e8f0"
MUTED = "#64748b"

CLASS_COLORS = [TEAL, AMBER, RED]
CLASS_NAMES  = ["Low", "Moderate", "High"]

#Signal definitions
SIGNALS = {
    "gsr":               {"label": "Galvanic Skin Response",       "weight": 0.20, "inverse": False},
    "task_switch":       {"label": "Cognitive Task-Switch Burden",  "weight": 0.18, "inverse": False},
    "voice_monotony":    {"label": "Voice Monotony Index",          "weight": 0.16, "inverse": False},
    "gait_irregularity": {"label": "Gait Irregularity",             "weight": 0.12, "inverse": False},
    "patient_rel":       {"label": "Patient Relationship Quality",  "weight": 0.12, "inverse": True},
    "color_chaos":       {"label": "Color-Pattern Chaos",           "weight": 0.12, "inverse": False},
    "tiktok_burnout":    {"label": "TikTok Burnout Signal",         "weight": 0.10, "inverse": False},
}

FEATURES = list(SIGNALS.keys())

#Strategy definitions
STRATEGIES = {
    "gsr":               {"method": "real_calibrated", "stressed_mean": 0.72, "neutral_mean": 0.28, "std": 0.09},
    "task_switch":       {"method": "synthetic", "slope": 0.75, "base": 0.10},
    "voice_monotony":    {"method": "real_calibrated", "stressed_mean": 0.68, "neutral_mean": 0.25, "std": 0.10},
    "gait_irregularity": {"method": "real_calibrated", "stressed_mean": 0.62, "neutral_mean": 0.22, "std": 0.08},
    "patient_rel":       {"method": "synthetic", "slope": 0.65, "base": 0.90, "inverse": True},
    "color_chaos":       {"method": "synthetic", "slope": 0.58, "base": 0.10},
    "tiktok_burnout":    {"method": "synthetic", "slope": 0.50, "base": 0.08},
}
