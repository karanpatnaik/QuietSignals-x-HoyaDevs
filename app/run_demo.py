from model.train import train_model
from model.predict import predict_abte

# run training
clf = train_model()

# manual test
predict_abte(
    clf,
    nurse_name="Cameron Lau",
    signal_values={
        "gsr": 0.82,
        "task_switch": 0.76,
        "voice_monotony": 0.71,
        "gait_irregularity": 0.68,
        "patient_rel": 0.22,
        "color_chaos": 0.74,
        "tiktok_burnout": 0.69,
    }
)
