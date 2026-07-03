"""Tests for the Iris classifier training pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import train_and_evaluate


def test_accuracy_at_least_090():
    """The decision tree should reach at least 90% accuracy on the test set."""
    accuracy = train_and_evaluate()
    assert accuracy >= 0.9, f"Accuracy {accuracy:.4f} is below the 0.9 threshold"
