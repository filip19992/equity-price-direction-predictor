from __future__ import annotations

from pathlib import Path


def build_default_config(project_root: Path) -> dict:
    """Shared experiment settings used by all modeling notebooks."""
    return {
        "dataset_path": project_root
        / "data"
        / "datasets"
        / "stock_panel_nine_tickers_session_aligned_full_history_rich_price_adjusted_google_score_raw.csv",
        "excluded_tickers": ["NFLX"],
        "neutral_band": 0.005,
        "test_size": 0.25,
        "validation_fraction_within_pretest": 0.25,
        "min_validation_dates": 20,
        "gap_days": 1,
        "random_state": 42,
        "selection_metric": "balanced_accuracy",
        "primary_validation_metric": "balanced_accuracy",
        "walk_forward_folds": 4,
        "walk_forward_validation_dates": 80,
        "walk_forward_min_train_dates": 252,
        "max_features_per_model": 14,
        "early_stopping_fraction": 0.2,
        "min_early_stopping_dates": 40,
        "min_fit_dates": 120,
    }
