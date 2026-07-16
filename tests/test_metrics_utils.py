"""Unit tests for pure-Python classification metrics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics_utils import (
    binary_classification_metrics,
    confusion_matrix_binary,
    precision_recall_f1,
)


def test_confusion_matrix_perfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    cm = confusion_matrix_binary(y_true, y_pred)
    assert cm == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_confusion_matrix_mixed():
    # true: 0,0,1,1  pred: 0,1,0,1 -> TN=1, FP=1, FN=1, TP=1
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    cm = confusion_matrix_binary(y_true, y_pred)
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_precision_recall_f1():
    cm = {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    prf = precision_recall_f1(cm)
    assert abs(prf["precision"] - 0.5) < 1e-9
    assert abs(prf["recall"] - 0.5) < 1e-9
    assert abs(prf["f1"] - 0.5) < 1e-9


def test_binary_classification_metrics_all_correct():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1]
    m = binary_classification_metrics(y_true, y_pred)
    assert abs(m["accuracy"] - 1.0) < 1e-9
    assert abs(m["precision"] - 1.0) < 1e-9
    assert abs(m["recall"] - 1.0) < 1e-9
    assert abs(m["f1"] - 1.0) < 1e-9
    assert m["n_samples"] == 5
    assert m["confusion_matrix"]["tp"] == 3
    assert m["confusion_matrix"]["tn"] == 2


def test_binary_classification_metrics_no_positives_predicted():
    y_true = [1, 1, 0]
    y_pred = [0, 0, 0]
    m = binary_classification_metrics(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["confusion_matrix"]["fn"] == 2
    assert m["confusion_matrix"]["tn"] == 1
