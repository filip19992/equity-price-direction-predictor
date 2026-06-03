from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "f1_score", "f1_weighted"]


class ClassificationMetrics:
    """Shared multiclass metric helpers for notebook reports."""

    CLASS_LABELS = {0: "down", 1: "neutral", 2: "up"}
    CLASS_VALUES = tuple(CLASS_LABELS)

    @staticmethod
    def metrics_from_predictions(y_true, preds) -> dict:
        y_true_array = np.asarray(y_true).astype(int)
        pred_array = np.asarray(preds).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true_array, pred_array)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_array, pred_array)),
            "f1_score": float(f1_score(y_true_array, pred_array, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true_array, pred_array, average="weighted", zero_division=0)),
            "preds": pred_array,
        }

    @classmethod
    def predictions_from_probabilities(cls, probabilities, classes=None) -> np.ndarray:
        probability_array = np.asarray(probabilities, dtype=float)
        if probability_array.ndim != 2:
            raise ValueError(f"Expected a 2D probability array, got shape={probability_array.shape}")

        class_values = np.asarray(cls.CLASS_VALUES if classes is None else classes).astype(int)
        if probability_array.shape[1] != len(class_values):
            raise ValueError(
                "Probability columns must match the number of classes: "
                f"probability_shape={probability_array.shape}, classes={class_values.tolist()}"
            )
        return class_values[np.argmax(probability_array, axis=1)].astype(int)

    @classmethod
    def probability_column_dict(cls, probabilities, classes=None, prefix: str = "probability") -> dict[str, np.ndarray]:
        probability_array = np.asarray(probabilities, dtype=float)
        class_values = np.asarray(cls.CLASS_VALUES if classes is None else classes).astype(int)
        if probability_array.shape[1] != len(class_values):
            raise ValueError(
                "Probability columns must match the number of classes: "
                f"probability_shape={probability_array.shape}, classes={class_values.tolist()}"
            )
        return {
            f"{prefix}_{cls.CLASS_LABELS.get(int(class_value), f'class_{int(class_value)}')}": probability_array[
                :, column_idx
            ]
            for column_idx, class_value in enumerate(class_values)
        }

