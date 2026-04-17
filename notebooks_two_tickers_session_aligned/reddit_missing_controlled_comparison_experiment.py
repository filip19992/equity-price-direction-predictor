from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from price_alt_baseline_experiment import (
    RANDOM_STATE,
    classification_metrics,
    engineer_ticker_features,
    get_feature_sets,
    locate_panel_dataset,
    rolling_zscore,
    split_train_test,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "reddit_missing_controlled_comparison"
TARGET_TICKERS = ["TSLA", "AAPL"]


MODEL_DEFS = {
    "logreg": Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "hist_gradient_boosting": Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=3,
                    max_iter=250,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "random_forest": Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=6,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


def current_series(frame: pd.DataFrame, column: str, fill: str | None = None) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce")
    if fill == "ffill":
        values = values.ffill()
    elif fill == "zero":
        values = values.fillna(0.0)
    return values


def average_available(series_list: list[pd.Series]) -> pd.Series:
    if not series_list:
        return pd.Series(dtype="float64")
    if len(series_list) == 1:
        return series_list[0]
    return pd.concat(series_list, axis=1).mean(axis=1)


def add_source_transforms(
    features: pd.DataFrame,
    source_name: str,
    source_series: dict[str, pd.Series],
) -> list[str]:
    created_cols: list[str] = []
    for var_name, series in source_series.items():
        prefix = f"{source_name}_{var_name}"
        values = pd.to_numeric(series, errors="coerce")
        features[f"{prefix}_lag1"] = values
        features[f"{prefix}_lag2"] = values.shift(1)
        features[f"{prefix}_roll3"] = values.rolling(3, min_periods=2).mean()
        features[f"{prefix}_roll5"] = values.rolling(5, min_periods=3).mean()
        features[f"{prefix}_surprise20"] = rolling_zscore(values, 20)
        created_cols.extend(
            [
                f"{prefix}_lag1",
                f"{prefix}_lag2",
                f"{prefix}_roll3",
                f"{prefix}_roll5",
                f"{prefix}_surprise20",
            ]
        )
    return created_cols


def build_presence_audit(frame: pd.DataFrame) -> dict[str, object]:
    subm_posts = current_series(frame, "subm_reddit_posts", fill="zero")
    comm_posts = current_series(frame, "comm_reddit_posts", fill="zero")
    subm_finbert = current_series(frame, "subm_reddit_finbert_weighted_mean")
    comm_finbert = current_series(frame, "comm_reddit_finbert_weighted_mean")

    any_posts = (subm_posts + comm_posts) > 0
    any_finbert = subm_finbert.notna() | comm_finbert.notna()

    return {
        "ticker": frame["ticker"].iloc[0],
        "rows": int(len(frame)),
        "pct_any_posts": float(any_posts.mean()),
        "pct_no_posts": float((~any_posts).mean()),
        "pct_subm_active": float((subm_posts > 0).mean()),
        "pct_comm_active": float((comm_posts > 0).mean()),
        "pct_subm_finbert_available": float(subm_finbert.notna().mean()),
        "pct_comm_finbert_available": float(comm_finbert.notna().mean()),
        "pct_any_finbert_available": float(any_finbert.mean()),
    }


def engineer_ticker_comparison_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, object]]:
    work = frame.sort_values("date").reset_index(drop=True).copy()
    base = engineer_ticker_features(work)
    price_cols = [col for col in get_feature_sets(base)["price_only"] if col != "ticker_code"]
    metadata_cols = ["date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir"]

    features = base[metadata_cols + price_cols].copy()
    feature_sets = {"price_only": list(price_cols)}

    subm_vader_legacy = current_series(work, "subm_reddit_vader_weighted_mean", fill="zero")
    comm_vader_legacy = current_series(work, "comm_reddit_vader_weighted_mean", fill="zero")
    subm_finbert_legacy = current_series(work, "subm_reddit_finbert_weighted_mean", fill="zero")
    comm_finbert_legacy = current_series(work, "comm_reddit_finbert_weighted_mean", fill="zero")
    reddit_vader_legacy = average_available([subm_vader_legacy, comm_vader_legacy])
    reddit_finbert_legacy = average_available([subm_finbert_legacy, comm_finbert_legacy])
    reddit_direction_legacy = average_available([reddit_vader_legacy, reddit_finbert_legacy])
    reddit_gap_legacy = (reddit_vader_legacy - reddit_finbert_legacy).abs()

    legacy_cols = add_source_transforms(
        features,
        "reddit_legacy",
        {
            "vader": reddit_vader_legacy,
            "finbert": reddit_finbert_legacy,
            "direction": reddit_direction_legacy,
            "gap": reddit_gap_legacy,
        },
    )
    feature_sets["price_plus_reddit_sentiment_legacy"] = list(
        dict.fromkeys(price_cols + legacy_cols)
    )

    subm_posts = current_series(work, "subm_reddit_posts", fill="zero")
    comm_posts = current_series(work, "comm_reddit_posts", fill="zero")
    subm_finbert = current_series(work, "subm_reddit_finbert_weighted_mean")
    comm_finbert = current_series(work, "comm_reddit_finbert_weighted_mean")

    features["reddit_presence_any_posts_flag"] = ((subm_posts + comm_posts) > 0).astype(float)
    features["reddit_presence_subm_active_flag"] = (subm_posts > 0).astype(float)
    features["reddit_presence_comm_active_flag"] = (comm_posts > 0).astype(float)
    features["reddit_presence_subm_finbert_available_flag"] = subm_finbert.notna().astype(float)
    features["reddit_presence_comm_finbert_available_flag"] = comm_finbert.notna().astype(float)

    controlled_cols = [
        "reddit_presence_any_posts_flag",
        "reddit_presence_subm_active_flag",
        "reddit_presence_comm_active_flag",
        "reddit_presence_subm_finbert_available_flag",
        "reddit_presence_comm_finbert_available_flag",
    ]
    feature_sets["price_plus_reddit_sentiment_legacy_plus_presence_flags"] = list(
        dict.fromkeys(price_cols + legacy_cols + controlled_cols)
    )

    audit_row = build_presence_audit(work)
    return features, feature_sets, audit_row


def load_panel(dataset_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in panel.columns:
        if col not in {"date", "ticker", "company_name", "y"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel


def build_feature_panel(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    pieces = []
    feature_sets_agg: dict[str, list[str]] = {}
    audit_rows: list[dict[str, object]] = []

    for _, ticker_frame in panel.groupby("ticker", sort=True):
        ticker_features, ticker_feature_sets, audit_row = engineer_ticker_comparison_features(
            ticker_frame
        )
        pieces.append(ticker_features)
        audit_rows.append(audit_row)
        for feature_set_name, cols in ticker_feature_sets.items():
            feature_sets_agg.setdefault(feature_set_name, [])
            feature_sets_agg[feature_set_name] = list(
                dict.fromkeys(feature_sets_agg[feature_set_name] + cols)
            )

    features = pd.concat(pieces, ignore_index=True)
    feature_cols = [
        col
        for col in features.columns
        if col not in {"date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir"}
    ]
    constant_cols = [col for col in feature_cols if features[col].nunique(dropna=True) <= 1]
    if constant_cols:
        features = features.drop(columns=constant_cols)
        feature_sets_agg = {
            name: [col for col in cols if col not in constant_cols]
            for name, cols in feature_sets_agg.items()
        }

    audit_df = pd.DataFrame(audit_rows).sort_values("ticker").reset_index(drop=True)
    return features, feature_sets_agg, audit_df


def build_test_delta_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    price_df = test_df[test_df["feature_set"] == "price_only"].copy()
    compare_df = test_df[test_df["feature_set"] != "price_only"].copy()
    merged = compare_df.merge(
        price_df[["ticker", "model_name", "accuracy", "balanced_accuracy", "roc_auc"]],
        on=["ticker", "model_name"],
        suffixes=("_variant", "_price"),
    )
    merged["delta_accuracy"] = merged["accuracy_variant"] - merged["accuracy_price"]
    merged["delta_balanced_accuracy"] = (
        merged["balanced_accuracy_variant"] - merged["balanced_accuracy_price"]
    )
    merged["delta_roc_auc"] = merged["roc_auc_variant"] - merged["roc_auc_price"]
    return merged[
        [
            "ticker",
            "model_name",
            "feature_set",
            "accuracy_variant",
            "accuracy_price",
            "delta_accuracy",
            "balanced_accuracy_variant",
            "balanced_accuracy_price",
            "delta_balanced_accuracy",
            "roc_auc_variant",
            "roc_auc_price",
            "delta_roc_auc",
        ]
    ].sort_values(["ticker", "model_name", "feature_set"]).reset_index(drop=True)


def run_reddit_missing_controlled_comparison(
    dataset_path: Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    tickers = tickers or list(TARGET_TICKERS)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(dataset_path)
    panel = panel[panel["ticker"].isin(tickers)].copy()
    feature_panel, feature_sets, audit_df = build_feature_panel(panel)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        ticker_frame = feature_panel[feature_panel["ticker"] == ticker].copy()
        train_df, test_df, _ = split_train_test(ticker_frame)

        for feature_set_name, feature_cols in feature_sets.items():
            for model_name, model_template in MODEL_DEFS.items():
                model = clone(model_template)

                train_work = train_df.dropna(subset=["y_dir"]).copy()
                test_work = test_df.dropna(subset=["y_dir"]).copy()

                X_train = train_work[feature_cols]
                y_train = train_work["y_dir"].astype(int)
                X_test = test_work[feature_cols]
                y_test = test_work["y_dir"].astype(int)

                model.fit(X_train, y_train)

                for split_name, split_frame, X_split, y_split in [
                    ("train", train_work, X_train, y_train),
                    ("test", test_work, X_test, y_test),
                ]:
                    proba = model.predict_proba(X_split)[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    metrics = classification_metrics(y_split, pred, proba)
                    metric_rows.append(
                        {
                            "ticker": ticker,
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "split": split_name,
                            "n_rows": int(len(split_frame)),
                            "n_features": int(len(feature_cols)),
                            **metrics,
                        }
                    )
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "date": split_frame["date"].values,
                                "ticker": split_frame["ticker"].values,
                                "split": split_name,
                                "model_name": model_name,
                                "feature_set": feature_set_name,
                                "future_return_1d": split_frame["future_return_1d"].values,
                                "y_true": y_split.values,
                                "y_pred": pred,
                                "p_up": proba,
                            }
                        )
                    )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["split", "ticker", "model_name", "feature_set"]
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["split", "ticker", "model_name", "feature_set", "date"]
    ).reset_index(drop=True)

    all_rows: list[dict[str, object]] = []
    for split_name in ["train", "test"]:
        split_pred = predictions_df[predictions_df["split"] == split_name].copy()
        for model_name in MODEL_DEFS:
            for feature_set_name, feature_cols in feature_sets.items():
                subset = split_pred[
                    (split_pred["model_name"] == model_name)
                    & (split_pred["feature_set"] == feature_set_name)
                ].copy()
                if subset.empty:
                    continue
                metrics = classification_metrics(
                    subset["y_true"],
                    subset["y_pred"].to_numpy(),
                    subset["p_up"].to_numpy(),
                )
                all_rows.append(
                    {
                        "ticker": "ALL",
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "split": split_name,
                        "n_rows": int(len(subset)),
                        "n_features": int(len(feature_cols)),
                        **metrics,
                    }
                )

    metrics_df = pd.concat([metrics_df, pd.DataFrame(all_rows)], ignore_index=True).sort_values(
        ["split", "ticker", "model_name", "feature_set"]
    ).reset_index(drop=True)
    delta_df = build_test_delta_summary(metrics_df)

    metrics_path = EXPERIMENT_DIR / "metrics.csv"
    predictions_path = EXPERIMENT_DIR / "predictions.csv"
    audit_path = EXPERIMENT_DIR / "presence_audit.csv"
    delta_path = EXPERIMENT_DIR / "test_deltas_vs_price_only.csv"
    feature_sets_path = EXPERIMENT_DIR / "feature_sets.json"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    feature_sets_path.write_text(json.dumps(feature_sets, indent=2, ensure_ascii=True), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "tickers": tickers,
                "models": list(MODEL_DEFS.keys()),
                "feature_sets": feature_sets,
                "comparison_goal": "legacy reddit sentiment features versus the same legacy features plus minimal presence flags",
                "legacy_definition": "reddit sentiment series are zero-filled before transform construction",
                "controlled_variant_definition": (
                    "the same legacy sentiment transform block plus only five current-session flags: "
                    "any_posts, subm_active, comm_active, subm_finbert_available, comm_finbert_available"
                ),
                "alt_time_alignment": "calendar-day alternative data aggregated to trading session and used as current-session features",
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    return {
        "dataset_path": dataset_path,
        "feature_panel": feature_panel,
        "feature_sets": feature_sets,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "presence_audit": audit_df,
        "test_deltas": delta_df,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "audit_path": audit_path,
        "delta_path": delta_path,
        "feature_sets_path": feature_sets_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    result = run_reddit_missing_controlled_comparison()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Audit: {result['audit_path']}")
    print(
        result["metrics"]
        .query("split == 'test'")
        .sort_values(["ticker", "model_name", "feature_set"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
