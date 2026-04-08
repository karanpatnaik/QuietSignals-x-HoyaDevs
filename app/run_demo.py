import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train import train_model
from model.predict import predict_abte

# run training
clf = train_model()

# Demo 1: all signals provided manually (behavioral + facial)
predict_abte(
    clf,
    nurse_name="Cameron Lau",
    signal_values={
        # behavioral signals
        "gsr":               0.82,
        "task_switch":       0.76,
        "voice_monotony":    0.71,
        "gait_irregularity": 0.68,
        "patient_rel":       0.22,
        "color_chaos":       0.74,
        "tiktok_burnout":    0.69,
        # facial signals (as if from grayscale analysis)
        "facial_negative_load":    0.38,
        "facial_flat_affect":      0.61,
        "facial_positive_protect": 0.15,
    }
)

# Demo 2: behavioral signals from Fitbit data; facial signals absent (neutral fallback)
predict_abte(
    clf,
    nurse_name="Fitbit Only Demo",
    signal_values={},
    fitbit_row={
        "hrv_rmssd":             28.0,
        "step_regularity":       0.45,
        "sleep_efficiency":      72.0,
        "sleep_duration_hrs":    5.5,
        "sedentary_bouts":       7,
        "sleep_onset_variability": 1.1,
        "n_awakenings":          5,
    }
)
