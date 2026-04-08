from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from model.generator import build_dataset
from model.signals import FEATURES

# Train QuietSignals model
def train_model():
    print("building dataset...")

    df = build_dataset(1500)

    X = df[FEATURES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    print("training model...")

    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1
    )

    clf.fit(X_train, y_train)

    acc = (clf.predict(X_test) == y_test).mean()
    print(f"done | accuracy: {acc:.3f}\n")

    return clf
