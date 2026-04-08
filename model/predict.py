import numpy as np
import matplotlib.pyplot as plt
import os

from model.signals import SIGNALS, FEATURES, CLASS_NAMES
from model.fitbit import fitbit_to_signals

# Neutral fallback for any signal not provided by the caller.
# Represents "no information" — contributes a mid-range value to the composite.
SIGNAL_NEUTRAL = 0.5


#Composite score
def composite_score(signal_dict):
    return round(sum(
        (1 - signal_dict.get(f, SIGNAL_NEUTRAL) if SIGNALS[f]["inverse"]
         else signal_dict.get(f, SIGNAL_NEUTRAL)) * SIGNALS[f]["weight"]
        for f in FEATURES
    ), 4)


#Predict function
def predict_abte(clf, signal_values: dict, nurse_name="Nurse",
                 fitbit_row=None, grayscale_report=None):
    """
    Parameters
    ----------
    clf              : trained sklearn classifier
    signal_values    : dict of known signal values (any subset of FEATURES)
    nurse_name       : label for output
    fitbit_row       : optional dict/row with Fitbit sensor data; fills the 7
                       behavioral signals via fitbit_to_signals()
    grayscale_report : optional dict returned by grayscale.analyze_image() or
                       analyze_stream(); fills the 3 facial signals
    """
    sv = signal_values.copy()

    # Merge Fitbit-derived behavioral signals
    if fitbit_row is not None:
        sv.update(fitbit_to_signals(fitbit_row))

    # Merge facial signals from grayscale analysis
    if grayscale_report is not None:
        from model.grayscale import grayscale_to_signals
        sv.update(grayscale_to_signals(grayscale_report))

    # Fill any still-missing signals with neutral value
    for f in FEATURES:
        sv.setdefault(f, SIGNAL_NEUTRAL)

    score = composite_score(sv)

    x = np.array([[sv[f] for f in FEATURES]])
    proba = clf.predict_proba(x)[0]
    label = CLASS_NAMES[int(np.argmax(proba))]

    print(f"{nurse_name}")
    print(f"score: {score:.4f}")
    print(f"risk : {label}\n")

    # save chart
    os.makedirs("outputs", exist_ok=True)

    plt.bar(CLASS_NAMES, proba)
    path = f"outputs/{nurse_name.replace(' ','_')}.png"
    plt.savefig(path)
    plt.close()

    print(f"chart saved → {path}\n")
