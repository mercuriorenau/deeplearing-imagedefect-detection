"""
Pure-Python classification metrics (no torch/dataset dependency).
Binary defect class is treated as the positive class (label == 1).
"""


def confusion_matrix_binary(y_true, y_pred, positive_label=1):
    """
    Return TN, FP, FN, TP for binary labels.
    y_true / y_pred: iterables of int labels.
    """
    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == positive_label and p == positive_label:
            tp += 1
        elif t == positive_label and p != positive_label:
            fn += 1
        elif t != positive_label and p == positive_label:
            fp += 1
        else:
            tn += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def precision_recall_f1(cm, eps=1e-12):
    """Compute precision, recall, F1 from a confusion-matrix dict."""
    tp = cm["tp"]
    fp = cm["fp"]
    fn = cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def binary_classification_metrics(y_true, y_pred, positive_label=1):
    """Full binary metrics package for the defect (positive) class."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    cm = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)
    prf = precision_recall_f1(cm)
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n else 0.0
    return {
        "accuracy": accuracy,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "confusion_matrix": cm,
        "n_samples": n,
    }
