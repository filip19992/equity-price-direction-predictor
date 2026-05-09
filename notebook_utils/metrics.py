from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "f1_score"]


class ClassificationMetrics:
    """Shared binary-classification metric helpers for notebook reports."""

    @staticmethod
    def metrics_from_scores(y_true: Iterable[int], scores: Iterable[float], threshold: float) -> dict:
        y_true_array = np.asarray(y_true).astype(int)
        score_array = np.asarray(scores, dtype=float)
        preds = (score_array >= threshold).astype(int)
        return {
            "accuracy": accuracy_score(y_true_array, preds),
            "balanced_accuracy": balanced_accuracy_score(y_true_array, preds),
            "f1_score": f1_score(y_true_array, preds, zero_division=0),
            "preds": preds,
        }

    @staticmethod
    def compute_from_predictions(prediction_frame: pd.DataFrame, metric_name: str) -> float:
        target_column = "y_true" if "y_true" in prediction_frame.columns else "target"
        y_true = prediction_frame[target_column].astype(int)
        y_pred = prediction_frame["prediction"].astype(int)
        if metric_name == "accuracy":
            return accuracy_score(y_true, y_pred)
        if metric_name == "balanced_accuracy":
            return balanced_accuracy_score(y_true, y_pred)
        if metric_name == "f1_score":
            return f1_score(y_true, y_pred, zero_division=0)
        raise ValueError(f"Unsupported bootstrap metric: {metric_name}")
