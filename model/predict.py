import numpy as np
import matplotlib.pyplot as plt
import os

from model.signals import SIGNALS, FEATURES, CLASS_NAMES
from model.fitbit import fitbit_to_signals

#Composite score
def composite_score(signal_dict):
    return round(sum(
        (1 - signal_dict[f] if SIGNALS[f]["inverse"] else signal_dict[f]) * SIGNALS[f]["weight"]
        for f in FEATURES
    ), 4)


#Predict function
def predict_abte(clf, signal_values: dict, nurse_name="Nurse", fitbit_row=None):

    # derive from fitbit if provided
    if fitbit_row is not None:
        signal_values = signal_values.copy()
        derived = fitbit_to_signals(fitbit_row)
        signal_values.update(derived)

    score = composite_score(signal_values)

    x = np.array([[signal_values[f] for f in FEATURES]])
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
