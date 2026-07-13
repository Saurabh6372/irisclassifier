"""Train and evaluate a Decision Tree classifier on the Iris dataset.

End-to-end workflow from the AI Fundamentals course lesson:
load data -> train/test split -> train -> predict -> evaluate.

Usage:
    python src/train.py

Prints the accuracy and classification report, saves the confusion matrix
figure to outputs/confusion_matrix.png, and exports the trained model to
outputs/iris_model.joblib (folder created automatically).

To reuse the trained model later without retraining:

    import joblib
    model = joblib.load("outputs/iris_model.joblib")
    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
"""

from pathlib import Path

import joblib

import matplotlib

matplotlib.use("Agg")  # headless-safe backend so the script runs anywhere
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "iris.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "iris_model.joblib"

TEST_SIZE = 0.2  # 80/20 train/test split -> 120 train / 30 test samples
RANDOM_STATE = 42  # deterministic shuffle so results are reproducible


def load_data():
    """Load the Iris dataset.

    Uses the local copy in data/iris.csv when available; otherwise loads it
    via scikit-learn and saves a local CSV copy for reproducibility.

    Returns:
        X, y, feature_names, target_names
    """
    iris = load_iris()
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded dataset from local copy: {DATA_PATH}")
    else:
        df = pd.DataFrame(iris.data, columns=feature_names)
        df["species"] = iris.target
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved a local copy of the dataset to {DATA_PATH}")

    X = df[feature_names].values
    y = df["species"].values
    return X, y, feature_names, target_names


def train_and_evaluate():
    """Run the full workflow and return the test-set accuracy."""
    # Step 1: prepare the data
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Step 2: choose and train the model
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Step 3: make predictions on unseen data
    y_pred = model.predict(X_test)
    print(f"First 5 predictions: {y_pred[:5].tolist()}")
    print(f"First 5 true labels: {y_test[:5].tolist()}")

    # Step 4: evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Feature importances:")
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {importance:.3f}")

    # Confusion matrix figure (outputs/ is created programmatically)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Iris Decision Tree — Confusion Matrix")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved to {out_path}")

    # Step 5: save the trained model so it can be reused without retraining
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")

    # Sanity check: reload the model and confirm it predicts identically
    reloaded = joblib.load(MODEL_PATH)
    assert (reloaded.predict(X_test) == y_pred).all()
    print("Reloaded model verified: predictions match the trained model")

    return accuracy


if __name__ == "__main__":
    train_and_evaluate()
