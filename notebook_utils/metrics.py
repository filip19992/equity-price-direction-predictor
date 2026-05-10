from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "f1_score"]


class ClassificationMetrics:
    """Shared binary-classification metric helpers for notebook reports."""

    @staticmethod
    def candidate_thresholds_from_scores(
        scores: Iterable[float],
        *,
        min_quantile: float,
        max_quantile: float,
        grid_size: int,
        default_threshold: float,
    ) -> np.ndarray:
        finite_scores = np.asarray(scores, dtype=float)
        finite_scores = finite_scores[np.isfinite(finite_scores)]
        if len(finite_scores) == 0:
            return np.array([default_threshold], dtype=float)

        quantiles = np.linspace(min_quantile, max_quantile, grid_size)
        thresholds = np.quantile(finite_scores, quantiles)
        return np.unique(np.r_[thresholds, default_threshold])

    @classmethod
    def best_threshold_for_balanced_accuracy(
        cls,
        y_true: Iterable[int],
        scores: Iterable[float],
        *,
        min_quantile: float,
        max_quantile: float,
        grid_size: int,
        default_threshold: float,
    ) -> tuple[float, float]:
        y_true_array = np.asarray(y_true).astype(int)
        score_array = np.asarray(scores, dtype=float)
        rows = []
        for threshold in cls.candidate_thresholds_from_scores(
            score_array,
            min_quantile=min_quantile,
            max_quantile=max_quantile,
            grid_size=grid_size,
            default_threshold=default_threshold,
        ):
            preds = (score_array >= threshold).astype(int)
            rows.append((float(threshold), float(balanced_accuracy_score(y_true_array, preds))))
        return max(rows, key=lambda item: item[1])

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

