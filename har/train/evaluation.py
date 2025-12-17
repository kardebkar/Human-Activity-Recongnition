from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    report: dict
    confusion: np.ndarray
    confusion_normalized: np.ndarray


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: list[str] | tuple[str, ...],
) -> ClassificationMetrics:
    cm = confusion_matrix(y_true, y_pred, labels=list(class_names))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(class_names),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    acc = float(accuracy_score(y_true, y_pred))

    return ClassificationMetrics(
        accuracy=acc,
        report=report,
        confusion=cm,
        confusion_normalized=cm_norm,
    )


def top_confusions(
    confusion: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    *,
    k: int = 5,
) -> list[dict[str, object]]:
    """Return the most common (actual -> predicted) mistakes."""

    class_names_list = list(class_names)
    pairs: list[dict[str, object]] = []
    for i, actual in enumerate(class_names_list):
        row_sum = float(confusion[i].sum()) if confusion[i].sum() else 0.0
        for j, predicted in enumerate(class_names_list):
            if i == j:
                continue
            count = int(confusion[i, j])
            if count <= 0:
                continue
            pct = (count / row_sum * 100.0) if row_sum else 0.0
            pairs.append({"actual": actual, "predicted": predicted, "count": count, "pct": pct})

    pairs.sort(key=lambda x: int(x["count"]), reverse=True)
    return pairs[:k]

