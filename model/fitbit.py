import numpy as np
import pandas as pd

# Fitbit signals from TILES-2018
FITBIT_SIGNALS = [
    "daily_steps", "step_regularity",
    "sedentary_minutes", "resting_hr",
    "hrv_rmssd", "sleep_duration_hrs",
    "sleep_efficiency"
]

# Fitbit data generator — mimics real structure
def generate_fitbit_data(n_nurses=212, n_weeks=10, seed=42):
    np.random.seed(seed)
    records = []

    for nurse_id in range(n_nurses):
        burnout = np.random.beta(2, 3)

        for week in range(n_weeks):
            for day in range(7):
                records.append({
                    "participant_id": f"id_{nurse_id}",
                    "week": week,
                    "daily_steps": np.random.normal(10000*(1-burnout), 2000),
                    "step_regularity": np.random.uniform(0.3, 0.9),
                    "sedentary_minutes": np.random.normal(400 + 300*burnout, 60),
                    "resting_hr": 60 + 15*burnout,
                    "hrv_rmssd": 50 - 25*burnout,
                    "sleep_duration_hrs": 7 - 2*burnout,
                    "sleep_efficiency": 90 - 20*burnout,
                })

    df = pd.DataFrame(records)

    # inject missing values
    for col in FITBIT_SIGNALS:
        mask = np.random.random(len(df)) < 0.1
        df.loc[mask, col] = np.nan

    return df


# Feature engineering
def engineer_fitbit_features(df):
    features = df.groupby("participant_id").mean(numeric_only=True).reset_index()
    return features


# Label generation
def fitbit_burnout_label(row):
    score = (
        (1 - min(row.get("hrv_rmssd", 40) / 60, 1)) * 0.5 +
        (min(row.get("resting_hr", 70), 90) - 55) / 35 * 0.5
    )
    return 0 if score < 0.35 else (1 if score < 0.60 else 2)


# Fitbit → QuietSignals bridge
def fitbit_to_signals(row):
    return {
        "gsr": 0.7,
        "task_switch": 0.6,
        "voice_monotony": 0.6,
        "gait_irregularity": 0.6,
        "patient_rel": 0.4,
        "color_chaos": 0.6,
        "tiktok_burnout": 0.6,
    }
