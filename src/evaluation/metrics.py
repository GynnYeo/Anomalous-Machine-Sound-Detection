"""Pure metric helpers for baseline binary classification evaluation."""

from __future__ import annotations

from typing import Sequence


def compute_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Return binary classification accuracy."""
    _validate_lengths(labels, predictions)
    if not labels:
        raise ValueError("Cannot compute accuracy for empty inputs.")
    correct = sum(int(label == prediction) for label, prediction in zip(labels, predictions))
    return correct / len(labels)


def compute_confusion_matrix(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> list[list[int]]:
    """Return the binary confusion matrix as [[tn, fp], [fn, tp]]."""
    _validate_lengths(labels, predictions)
    tn = fp = fn = tp = 0
    for label, prediction in zip(labels, predictions):
        if label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1 and prediction == 0:
            fn += 1
        elif label == 1 and prediction == 1:
            tp += 1
        else:
            raise ValueError(
                f"Expected binary labels/predictions in {{0, 1}}, got label={label}, prediction={prediction}."
            )
    return [[tn, fp], [fn, tp]]


def compute_precision(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Return binary precision for the positive class."""
    _, fp, _, tp = _flatten_confusion_matrix(compute_confusion_matrix(labels, predictions))
    denominator = tp + fp
    if denominator == 0:
        return 0.0
    return tp / denominator


def compute_recall(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Return binary recall for the positive class."""
    _, _, fn, tp = _flatten_confusion_matrix(compute_confusion_matrix(labels, predictions))
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    return tp / denominator


def compute_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Return binary F1 score for the positive class."""
    precision = compute_precision(labels, predictions)
    recall = compute_recall(labels, predictions)
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2.0 * precision * recall / denominator


def compute_roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Return binary ROC-AUC from positive-class probabilities."""
    _validate_lengths(labels, probabilities)
    if not labels:
        raise ValueError("Cannot compute ROC-AUC for empty inputs.")

    positives = sum(int(label == 1) for label in labels)
    negatives = sum(int(label == 0) for label in labels)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both positive and negative samples.")

    ranked_pairs = sorted(
        zip(probabilities, labels),
        key=lambda item: item[0],
    )

    rank_sum = 0.0
    index = 0
    while index < len(ranked_pairs):
        tie_start = index
        tie_value = ranked_pairs[index][0]
        while index < len(ranked_pairs) and ranked_pairs[index][0] == tie_value:
            index += 1
        average_rank = (tie_start + 1 + index) / 2.0
        positive_count = sum(label == 1 for _, label in ranked_pairs[tie_start:index])
        rank_sum += average_rank * positive_count

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def compute_binary_classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[float] | None = None,
) -> dict[str, float | list[list[int]]]:
    """Return a compact baseline metric dictionary for binary classification."""
    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": compute_accuracy(labels, predictions),
        "precision": compute_precision(labels, predictions),
        "recall": compute_recall(labels, predictions),
        "f1": compute_f1(labels, predictions),
        "confusion_matrix": compute_confusion_matrix(labels, predictions),
    }
    if probabilities is not None:
        try:
            metrics["roc_auc"] = compute_roc_auc(labels, probabilities)
        except ValueError:
            pass
    return metrics


def _validate_lengths(first: Sequence[object], second: Sequence[object]) -> None:
    """Ensure paired metric inputs have matching lengths."""
    if len(first) != len(second):
        raise ValueError(
            f"Metric inputs must have the same length, got {len(first)} and {len(second)}."
        )


def _flatten_confusion_matrix(confusion_matrix: list[list[int]]) -> tuple[int, int, int, int]:
    """Flatten [[tn, fp], [fn, tp]] into a tuple."""
    return (
        confusion_matrix[0][0],
        confusion_matrix[0][1],
        confusion_matrix[1][0],
        confusion_matrix[1][1],
    )
