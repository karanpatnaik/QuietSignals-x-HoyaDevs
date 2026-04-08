import numpy as np
import pandas as pd

from model.signals import SIGNALS, FEATURES, STRATEGIES

#Data generation (mix of synthetic + real data)

def generate_signal(latent, strategy, noise=0.08):
    """
    latent  : float 0-1, underlying fatigue level
    strategy: dict with generation parameters
    """
    method = strategy["method"]

    if method == "real_calibrated":
        # Calibrated from published mean/std of real datasets.
        stressed_mean = strategy["stressed_mean"]
        neutral_mean  = strategy["neutral_mean"]
        std           = strategy["std"]
        raw = neutral_mean + latent * (stressed_mean - neutral_mean)
        raw += np.random.normal(0, std)
        return np.clip(raw, 0, 1)

    elif method == "synthetic":
        # No real dataset — logic derived from paper methodology
        slope = strategy["slope"]
        base  = strategy.get("base", 0.0)
        inv   = strategy.get("inverse", False)
        raw   = base + (slope * latent if not inv else slope * (1 - latent))
        raw  += np.random.normal(0, noise)
        return np.clip(raw, 0, 1)


# Build QuietSignals training dataset
def build_dataset(n=1500):
    """
    Latent fatigue drawn from Beta(1.4, 2.0)
    """
    np.random.seed(42)
    rows = []

    for _ in range(n):
        latent = np.random.beta(1.4, 2.0)
        vals = {f: generate_signal(latent, STRATEGIES[f]) for f in FEATURES}

        score = sum(
            (1 - vals[f] if SIGNALS[f]["inverse"] else vals[f]) * SIGNALS[f]["weight"]
            for f in FEATURES
        )

        label = 0 if score < 0.30 else (1 if score < 0.50 else 2)
        rows.append([vals[f] for f in FEATURES] + [round(score, 4), label])

    return pd.DataFrame(rows, columns=FEATURES + ["composite_score", "label"])
