"""Tests for the Iris classifier training pipeline."""

import sys
from pathlib import Path

import joblib
from sklearn.datasets import load_iris

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import MODEL_PATH, train_and_evaluate


def test_accuracy_at_least_090():
    """The decision tree should reach at least 90% accuracy on the test set."""
    accuracy = train_and_evaluate()
    assert accuracy >= 0.9, f"Accuracy {accuracy:.4f} is below the 0.9 threshold"


def test_saved_model_loads_and_predicts():
    """The exported model must load from disk and predict without retraining."""
    train_and_evaluate()
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"

    model = joblib.load(MODEL_PATH)
    iris = load_iris()
    predictions = model.predict(iris.data[:5])
    assert list(predictions) == list(iris.target[:5])
