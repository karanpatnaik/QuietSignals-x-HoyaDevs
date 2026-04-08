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
# Maps Fitbit sensor features to the 7 behavioral QuietSignals signals.
# Facial signals (facial_negative_load, facial_flat_affect, facial_positive_protect)
# are not derivable from Fitbit — they come from grayscale.py instead.
def fitbit_to_signals(row):
    def get(key, default):
        v = row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)
        return default if (v is None or (isinstance(v, float) and v != v)) else v

    def clip(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))

    # gsr: low HRV → high sympathetic activation → high skin conductance proxy
    # hrv_rmssd range: ~10 ms (stressed) to ~55 ms (healthy)
    hrv = get("hrv_rmssd", 35)
    gsr = clip(1.0 - (hrv - 10) / 45)

    # gait_irregularity: direct inverse of step regularity
    # step_regularity range: 0.1 (erratic) to 0.8 (consistent)
    step_reg = get("step_regularity", 0.5)
    gait_irregularity = clip(1.0 - (step_reg - 0.1) / 0.7)

    # voice_monotony: poor sleep efficiency → flat, monotone speech
    # sleep_efficiency range: 40% (very poor) to 98% (excellent)
    sleep_eff = get("sleep_efficiency", 80)
    voice_monotony = clip(1.0 - (sleep_eff - 40) / 58)

    # patient_rel: restorative sleep supports interpersonal quality (inverse signal)
    # sleep_duration range: 2 hrs (severe deprivation) to 7.2 hrs (healthy)
    sleep_hrs = get("sleep_duration_hrs", 6)
    patient_rel = clip((sleep_hrs - 2) / 5.2)

    # task_switch: fragmented movement = fragmented cognitive attention
    # sedentary_bouts range: 3 (active) to 11 (very sedentary/fragmented)
    # Falls back to resting_hr if sedentary_bouts not available
    sed_bouts = get("sedentary_bouts", None)
    if sed_bouts is not None:
        task_switch = clip((sed_bouts - 3) / 8)
    else:
        rhr = get("resting_hr", 70)
        task_switch = clip((rhr - 58) / 18)

    # color_chaos: irregular sleep schedule → cognitive/perceptual chaos
    # sleep_onset_variability range: 0.1 (regular) to 2.0 (highly irregular)
    sov = get("sleep_onset_variability", None)
    if sov is not None:
        color_chaos = clip((sov - 0.1) / 1.9)
    else:
        # fallback: excess awake time during sleep window
        awake = get("awake_min", 20)
        color_chaos = clip((awake - 15) / 25)

    # tiktok_burnout: night awakenings proxy for late-night phone use / poor sleep hygiene
    # n_awakenings range: 2 (healthy) to 9 (severely disrupted)
    n_awk = get("n_awakenings", None)
    if n_awk is not None:
        tiktok_burnout = clip((n_awk - 2) / 7)
    else:
        awake = get("awake_min", 20)
        tiktok_burnout = clip((awake - 15) / 25)

    return {
        "gsr":               round(gsr, 4),
        "task_switch":       round(task_switch, 4),
        "voice_monotony":    round(voice_monotony, 4),
        "gait_irregularity": round(gait_irregularity, 4),
        "patient_rel":       round(patient_rel, 4),
        "color_chaos":       round(color_chaos, 4),
        "tiktok_burnout":    round(tiktok_burnout, 4),
    }
