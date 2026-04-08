import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.train import train_model
from model.predict import predict_abte

clf = train_model()

predict_abte(
    clf,
    nurse_name=os.environ.get("NURSE_NAME", "Cameron Lau"),
    signal_values={
        "gsr":               float(os.environ.get("GSR", 0.82)),
        "task_switch":       float(os.environ.get("TASK_SWITCH", 0.76)),
        "voice_monotony":    float(os.environ.get("VOICE_MONOTONY", 0.71)),
        "gait_irregularity": float(os.environ.get("GAIT_IRREGULARITY", 0.68)),
        "patient_rel":       float(os.environ.get("PATIENT_REL", 0.22)),
        "color_chaos":       float(os.environ.get("COLOR_CHAOS", 0.74)),
        "tiktok_burnout":    float(os.environ.get("TIKTOK_BURNOUT", 0.69)),
    }
)
