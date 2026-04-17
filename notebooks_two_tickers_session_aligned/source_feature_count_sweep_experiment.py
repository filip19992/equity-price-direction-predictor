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
EXPERIMENT_DIR = OUTPUT_DIR / "source_feature_count_sweep_experiment"
TARGET_TICKERS = ["TSLA", "AAPL"]

PRICE_COUNTS = [5, 10, 14]
GOOGLE_COUNTS = [1, 2, 3]
GDELT_COUNTS = [2, 4, 6]
REDDIT_COUNTS = [2, 4, 7]

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

PRICE_FEATURE_ORDER = [
    "ret_1d",
    "ret_5d",
    "realized_vol_5d",
    "price_vs_sma_5d",
    "volume_change_1d",
    "ret_2d",
    "ret_10d",
    "realized_vol_10d",
    "price_vs_sma_10d",
    "volume_z20",
    "volume_change_5d",
    "volume_vs_sma_20d",
    "weekday_sin",
    "weekday_cos",
]

GOOGLE_FEATURE_ORDER = [
    "trends_level_z20",
    "trends_change_1d",
    "trends_change_7d",
]

GDELT_FEATURE_ORDER = [
    "gdelt_articles_z20",
    "gdelt_articles_change_1d",
    "gdelt_sentiment_5d",
    "gdelt_sentiment_surprise20",
    "gdelt_robust_current",
    "gdelt_robust_roll3",
]

REDDIT_FEATURE_ORDER = [
    "subm_direction_roll3",
    "comm_direction_roll3",
    "reddit_attention_z20_small",
    "subm_direction_surprise20",
    "comm_direction_surprise20",
    "subm_active_flag_small",
    "comm_active_flag_small",
]


def current_series(frame: pd.DataFrame, column: str, fill: str | None = None) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce")
    if fill == "zero":
        values = values.fillna(0.0)
    elif fill == "ffill":
        values = values.ffill()
    return values


def average_available(series_list: list[pd.Series]) -> pd.Series:
    valid = [series for series in series_list if series is not None]
    if not valid:
        return pd.Series(dtype="float64")
    if len(valid) == 1:
        return valid[0]
    return pd.concat(valid, axis=1).mean(axis=1)


def load_panel(dataset_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in panel.columns:
        if col not in {"date", "ticker", "company_name", "y"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel


def engineer_sweep_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    work = frame.sort_values("date").reset_index(drop=True).copy()
    base = engineer_ticker_features(work)

    metadata_cols = ["date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir"]
    available = set(base.columns)
    price_cols = [col for col in PRICE_FEATURE_ORDER if col in available]

    features = base[metadata_cols + price_cols].copy()

    google_cols = [col for col in GOOGLE_FEATURE_ORDER if col in available]
    for col in google_cols:
        features[col] = base[col]

    gdelt_robust = current_series(work, "gdelt_robust", fill="ffill")
    features["gdelt_sentiment_surprise20"] = rolling_zscore(current_series(work, "gdelt_sentiment_score"), 20)
    features["gdelt_robust_current"] = gdelt_robust
    features["gdelt_robust_roll3"] = gdelt_robust.rolling(3, min_periods=2).mean()
    gdelt_cols = [col for col in GDELT_FEATURE_ORDER if col in features.columns]

    subm_posts = current_series(work, "subm_reddit_posts", fill="zero")
    comm_posts = current_series(work, "comm_reddit_posts", fill="zero")
    subm_vader = current_series(work, "subm_reddit_vader_weighted_mean")
    comm_vader = current_series(work, "comm_reddit_vader_weighted_mean")
    subm_finbert = current_series(work, "subm_reddit_finbert_weighted_mean")
    comm_finbert = current_series(work, "comm_reddit_finbert_weighted_mean")

    subm_direction = average_available([subm_vader, subm_finbert])
    comm_direction = average_available([comm_vader, comm_finbert])
    total_posts_log = np.log1p((subm_posts + comm_posts).clip(lower=0.0))

    features["subm_active_flag_small"] = (subm_posts > 0).astype(float)
    features["comm_active_flag_small"] = (comm_posts > 0).astype(float)
    features["subm_direction_roll3"] = subm_direction.rolling(3, min_periods=2).mean()
    features["subm_direction_surprise20"] = rolling_zscore(subm_direction, 20)
    features["comm_direction_roll3"] = comm_direction.rolling(3, min_periods=2).mean()
    features["comm_direction_surprise20"] = rolling_zscore(comm_direction, 20)
    features["reddit_attention_z20_small"] = rolling_zscore(total_posts_log, 20)
    reddit_cols = [col for col in REDDIT_FEATURE_ORDER if col in features.columns]

    family_orders = {
        "price": price_cols,
        "google": google_cols,
        "gdelt": gdelt_cols,
        "reddit": reddit_cols,
    }

    feature_sets: dict[str, list[str]] = {}
    for count in PRICE_COUNTS:
        cols = price_cols[: min(count, len(price_cols))]
        feature_sets[f"price_top{len(cols)}"] = cols

    base_price = price_cols
    for count in GOOGLE_COUNTS:
        cols = google_cols[: min(count, len(google_cols))]
        feature_sets[f"price14_plus_google_top{len(cols)}"] = list(dict.fromkeys(base_price + cols))
    for count in GDELT_COUNTS:
        cols = gdelt_cols[: min(count, len(gdelt_cols))]
        feature_sets[f"price14_plus_gdelt_top{len(cols)}"] = list(dict.fromkeys(base_price + cols))
    for count in REDDIT_COUNTS:
        cols = reddit_cols[: min(count, len(reddit_cols))]
        feature_sets[f"price14_plus_reddit_top{len(cols)}"] = list(dict.fromkeys(base_price + cols))

    return features, feature_sets, family_orders


def parse_feature_set_name(feature_set_name: str) -> dict[str, object]:
    if feature_set_name.startswith("price_top"):
        count = int(feature_set_name.replace("price_top", ""))
        return {
            "source_family": "price",
            "price_feature_count": count,
            "source_feature_count": 0,
        }
    if "plus_google_top" in feature_set_name:
        count = int(feature_set_name.split("plus_google_top", 1)[1])
        return {
            "source_family": "google",
            "price_feature_count": 14,
            "source_feature_count": count,
        }
    if "plus_gdelt_top" in feature_set_name:
        count = int(feature_set_name.split("plus_gdelt_top", 1)[1])
        return {
            "source_family": "gdelt",
            "price_feature_count": 14,
            "source_feature_count": count,
        }
    if "plus_reddit_top" in feature_set_name:
        count = int(feature_set_name.split("plus_reddit_top", 1)[1])
        return {
            "source_family": "reddit",
            "price_feature_count": 14,
            "source_feature_count": count,
        }
    return {
        "source_family": "unknown",
        "price_feature_count": np.nan,
        "source_feature_count": np.nan,
    }


def build_feature_panel(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    pieces = []
    feature_sets_agg: dict[str, list[str]] = {}
    family_orders: dict[str, list[str]] = {}

    for _, ticker_frame in panel.groupby("ticker", sort=True):
        ticker_features, ticker_feature_sets, ticker_family_orders = engineer_sweep_features(ticker_frame)
        pieces.append(ticker_features)
        for name, cols in ticker_feature_sets.items():
            feature_sets_agg.setdefault(name, [])
            feature_sets_agg[name] = list(dict.fromkeys(feature_sets_agg[name] + cols))
        for family, cols in ticker_family_orders.items():
            family_orders.setdefault(family, [])
            family_orders[family] = list(dict.fromkeys(family_orders[family] + cols))

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
        family_orders = {
            family: [col for col in cols if col not in constant_cols]
            for family, cols in family_orders.items()
        }

    return features, feature_sets_agg, family_orders


def build_test_delta_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    price14_df = test_df[test_df["feature_set"] == "price_top14"].copy()
    compare_df = test_df[test_df["feature_set"] != "price_top14"].copy()
    merged = compare_df.merge(
        price14_df[["ticker", "model_name", "accuracy", "balanced_accuracy", "roc_auc"]],
        on=["ticker", "model_name"],
        suffixes=("_variant", "_price14"),
    )
    merged["delta_accuracy_vs_price14"] = merged["accuracy_variant"] - merged["accuracy_price14"]
    merged["delta_balanced_accuracy_vs_price14"] = (
        merged["balanced_accuracy_variant"] - merged["balanced_accuracy_price14"]
    )
    merged["delta_roc_auc_vs_price14"] = merged["roc_auc_variant"] - merged["roc_auc_price14"]
    return merged.sort_values(["ticker", "model_name", "feature_set"]).reset_index(drop=True)


def build_best_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    best_rows = []
    for (ticker, model_name, source_family), group in test_df.groupby(["ticker", "model_name", "source_family"]):
        best = group.sort_values(["balanced_accuracy", "roc_auc"], ascending=[False, False]).iloc[0]
        best_rows.append(best.to_dict())
    return pd.DataFrame(best_rows).sort_values(
        ["ticker", "model_name", "source_family"]
    ).reset_index(drop=True)


def run_source_feature_count_sweep_experiment(
    dataset_path: Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    tickers = tickers or list(TARGET_TICKERS)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(dataset_path)
    panel = panel[panel["ticker"].isin(tickers)].copy()
    feature_panel, feature_sets, family_orders = build_feature_panel(panel)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        ticker_frame = feature_panel[feature_panel["ticker"] == ticker].copy()
        train_df, test_df, _ = split_train_test(ticker_frame)

        for feature_set_name, feature_cols in feature_sets.items():
            feature_meta = parse_feature_set_name(feature_set_name)
            train_work = train_df.dropna(subset=["y_dir"]).copy()
            test_work = test_df.dropna(subset=["y_dir"]).copy()
            train_rows_per_feature = len(train_work) / max(len(feature_cols), 1)

            X_train = train_work[feature_cols]
            y_train = train_work["y_dir"].astype(int)
            X_test = test_work[feature_cols]
            y_test = test_work["y_dir"].astype(int)

            for model_name, model_template in MODEL_DEFS.items():
                model = clone(model_template)
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
                            "train_rows_per_feature": float(train_rows_per_feature),
                            **feature_meta,
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
        ["split", "ticker", "model_name", "source_family", "feature_set"]
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
                feature_meta = parse_feature_set_name(feature_set_name)
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
                        "train_rows_per_feature": np.nan,
                        **feature_meta,
                        **metrics,
                    }
                )

    metrics_df = pd.concat([metrics_df, pd.DataFrame(all_rows)], ignore_index=True).sort_values(
        ["split", "ticker", "model_name", "source_family", "feature_set"]
    ).reset_index(drop=True)
    delta_df = build_test_delta_summary(metrics_df)
    best_df = build_best_summary(metrics_df)

    metrics_path = EXPERIMENT_DIR / "metrics.csv"
    predictions_path = EXPERIMENT_DIR / "predictions.csv"
    delta_path = EXPERIMENT_DIR / "test_deltas_vs_price14.csv"
    best_path = EXPERIMENT_DIR / "best_by_source.csv"
    feature_sets_path = EXPERIMENT_DIR / "feature_sets.json"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    best_df.to_csv(best_path, index=False)
    feature_sets_path.write_text(json.dumps(feature_sets, indent=2, ensure_ascii=True), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "tickers": tickers,
                "models": list(MODEL_DEFS.keys()),
                "feature_sets": feature_sets,
                "family_orders": family_orders,
                "comparison_goal": "sweep source-specific top-k feature counts for price, Google, GDELT and Reddit",
                "count_grid": {
                    "price": PRICE_COUNTS,
                    "google": GOOGLE_COUNTS,
                    "gdelt": GDELT_COUNTS,
                    "reddit": REDDIT_COUNTS,
                },
                "selection_rule": "fixed heuristic feature order per source, not tuned on test",
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
        "family_orders": family_orders,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "test_deltas": delta_df,
        "best_by_source": best_df,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "delta_path": delta_path,
        "best_path": best_path,
        "feature_sets_path": feature_sets_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    result = run_source_feature_count_sweep_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Best summary: {result['best_path']}")
    print(
        result["best_by_source"]
        .sort_values(["ticker", "model_name", "source_family"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
