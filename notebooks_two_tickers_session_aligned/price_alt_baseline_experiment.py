from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
DATASET_DIR = OUTPUT_DIR / "datasets"
EXPERIMENT_DIR = OUTPUT_DIR / "price_alt_baseline"
DEFAULT_DATASET_NAME = "stock_direction_two_tickers_session_aligned_base.csv"

RANDOM_STATE = 42
FINAL_HOLDOUT_MONTHS = 6


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


def locate_panel_dataset(dataset_name: str = DEFAULT_DATASET_NAME) -> Path:
    path = DATASET_DIR / dataset_name
    if path.exists():
        return path
    raise FileNotFoundError(f"Session-aligned dataset not found: {path}")


def rolling_zscore(series: pd.Series, window: int = 20, min_periods: int = 10) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    values = series.pct_change(periods)
    return values.replace([np.inf, -np.inf], np.nan)


def signed_log1p(series: pd.Series) -> pd.Series:
    return np.sign(series) * np.log1p(np.abs(series))


def shifted_series(frame: pd.DataFrame, column: str, fill: str | None = None) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce").shift(1)
    if fill == "ffill":
        values = values.ffill()
    elif fill == "zero":
        values = values.fillna(0.0)
    return values


def add_reddit_block(frame: pd.DataFrame, features: pd.DataFrame, prefix: str) -> None:
    posts_col = f"{prefix}reddit_posts"
    score_col = f"{prefix}reddit_score_sum"
    comments_col = f"{prefix}reddit_comments_sum"
    vader_col = (
        f"{prefix}reddit_vader_weighted_mean"
        if f"{prefix}reddit_vader_weighted_mean" in frame.columns
        else f"{prefix}reddit_vader_mean"
    )
    finbert_col = (
        f"{prefix}reddit_finbert_weighted_mean"
        if f"{prefix}reddit_finbert_weighted_mean" in frame.columns
        else f"{prefix}reddit_finbert_mean"
    )

    if posts_col in frame.columns:
        posts = pd.to_numeric(frame[posts_col], errors="coerce").fillna(0.0)
        log_posts = np.log1p(posts.clip(lower=0))
        features[f"{prefix}posts_z20"] = rolling_zscore(log_posts, 20)
        features[f"{prefix}active_flag"] = (posts > 0).astype(float)

    if score_col in frame.columns:
        score = pd.to_numeric(frame[score_col], errors="coerce").fillna(0.0)
        features[f"{prefix}score_signed_log"] = signed_log1p(score)

    if comments_col in frame.columns:
        comments = pd.to_numeric(frame[comments_col], errors="coerce").fillna(0.0)
        features[f"{prefix}comments_log"] = np.log1p(comments.clip(lower=0))

    if vader_col in frame.columns:
        features[f"{prefix}vader_mean"] = pd.to_numeric(frame[vader_col], errors="coerce")

    if finbert_col in frame.columns:
        features[f"{prefix}finbert_mean"] = pd.to_numeric(frame[finbert_col], errors="coerce")


def engineer_ticker_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_values("date").reset_index(drop=True).copy()

    features = pd.DataFrame(
        {
            "date": work["date"],
            "ticker": work["ticker"],
            "company_name": work["company_name"],
            "future_return_1d": work["future_return_1d"],
            "y": work["y"],
            "y_code": work["y_code"],
            "y_dir": np.where(work["future_return_1d"].isna(), np.nan, (work["future_return_1d"] > 0).astype(float)),
            "ticker_code": float(work["ticker"].iloc[0] == "AAPL"),
        }
    )

    ret_1d = pd.to_numeric(work["stock_return_1d"], errors="coerce")
    features["ret_1d"] = ret_1d
    features["ret_2d"] = safe_pct_change(work["stock_price"], 2)
    features["ret_5d"] = pd.to_numeric(work["stock_return_5d"], errors="coerce")
    features["ret_10d"] = safe_pct_change(work["stock_price"], 10)
    features["volume_change_1d"] = pd.to_numeric(work["stock_volume_change_1d"], errors="coerce")
    features["volume_change_5d"] = safe_pct_change(work["stock_volume"], 5)
    features["realized_vol_5d"] = ret_1d.rolling(5).std()
    features["realized_vol_10d"] = ret_1d.rolling(10).std()
    features["price_vs_sma_5d"] = work["stock_price"] / work["stock_price"].rolling(5).mean() - 1
    features["price_vs_sma_10d"] = work["stock_price"] / work["stock_price"].rolling(10).mean() - 1

    log_volume = np.log1p(work["stock_volume"].clip(lower=0))
    volume_sma20 = work["stock_volume"].rolling(20, min_periods=10).mean()
    features["volume_z20"] = rolling_zscore(log_volume, 20)
    features["volume_vs_sma_20d"] = work["stock_volume"] / volume_sma20 - 1

    weekday = work["date"].dt.dayofweek.astype(float)
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 5.0)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 5.0)

    trends_level = pd.to_numeric(work["google_trends_score"], errors="coerce")
    features["trends_level_z20"] = rolling_zscore(trends_level, 20)
    features["trends_change_1d"] = safe_pct_change(trends_level, 1).clip(-1.0, 1.0)
    features["trends_change_7d"] = safe_pct_change(trends_level, 7).clip(-1.0, 1.0)

    gdelt_articles = pd.to_numeric(work["gdelt_articles"], errors="coerce")
    features["gdelt_articles_z20"] = rolling_zscore(np.log1p(gdelt_articles.clip(lower=0)), 20)
    features["gdelt_articles_change_1d"] = safe_pct_change(gdelt_articles, 1).clip(-3.0, 3.0)

    gdelt_sentiment = pd.to_numeric(work["gdelt_sentiment_score"], errors="coerce")
    features["gdelt_sentiment_score"] = gdelt_sentiment
    features["gdelt_sentiment_5d"] = gdelt_sentiment.rolling(5, min_periods=3).mean()

    add_reddit_block(work, features, "subm_")
    add_reddit_block(work, features, "comm_")

    return features


def build_feature_panel(panel: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, ticker_frame in panel.groupby("ticker", sort=True):
        pieces.append(engineer_ticker_features(ticker_frame))
    features = pd.concat(pieces, ignore_index=True)

    feature_cols = [
        col
        for col in features.columns
        if col not in ["date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir"]
    ]
    constant_cols = [col for col in feature_cols if features[col].nunique(dropna=True) <= 1]
    if constant_cols:
        features = features.drop(columns=constant_cols)

    return features


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        result["roc_auc"] = np.nan
    return result


def get_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    available = set(frame.columns)
    price_features = [
        "ticker_code",
        "ret_1d",
        "ret_2d",
        "ret_5d",
        "ret_10d",
        "volume_change_1d",
        "volume_change_5d",
        "realized_vol_5d",
        "realized_vol_10d",
        "price_vs_sma_5d",
        "price_vs_sma_10d",
        "volume_z20",
        "volume_vs_sma_20d",
        "weekday_sin",
        "weekday_cos",
    ]
    alt_features = [
        "ticker_code",
        "trends_level_z20",
        "trends_change_1d",
        "trends_change_7d",
        "gdelt_articles_z20",
        "gdelt_articles_change_1d",
        "gdelt_sentiment_score",
        "gdelt_sentiment_5d",
        "subm_posts_z20",
        "subm_active_flag",
        "subm_score_signed_log",
        "subm_comments_log",
        "subm_vader_mean",
        "subm_finbert_mean",
        "comm_posts_z20",
        "comm_active_flag",
        "comm_score_signed_log",
        "comm_comments_log",
        "comm_vader_mean",
        "comm_finbert_mean",
    ]

    feature_sets = {
        "price_only": [col for col in price_features if col in available],
        "alt_only": [col for col in alt_features if col in available],
    }
    feature_sets["price_plus_alt"] = list(dict.fromkeys(feature_sets["price_only"] + feature_sets["alt_only"]))
    return feature_sets


def split_train_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    max_date = frame["date"].max().normalize()
    test_start = (max_date - pd.DateOffset(months=FINAL_HOLDOUT_MONTHS)).normalize()
    train_df = frame[frame["date"] < test_start].copy()
    test_df = frame[frame["date"] >= test_start].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty.")
    return train_df, test_df, test_start


def run_baseline_experiment(dataset_path: Path | None = None) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in panel.columns:
        if col not in {"date", "ticker", "company_name", "y"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    feature_panel = build_feature_panel(panel)
    train_df, test_df, test_start = split_train_test(feature_panel)
    feature_sets = get_feature_sets(feature_panel)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

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
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "split": split_name,
                        "ticker": "ALL",
                        "n_rows": int(len(split_frame)),
                        "n_features": int(len(feature_cols)),
                        **metrics,
                    }
                )

                for ticker, ticker_frame in split_frame.groupby("ticker"):
                    ticker_idx = ticker_frame.index
                    pred_slice = pred[split_frame.index.get_indexer(ticker_idx)]
                    proba_slice = proba[split_frame.index.get_indexer(ticker_idx)]
                    y_slice = y_split.loc[ticker_idx]
                    ticker_metrics = classification_metrics(y_slice, pred_slice, proba_slice)
                    metric_rows.append(
                        {
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "split": split_name,
                            "ticker": ticker,
                            "n_rows": int(len(ticker_frame)),
                            "n_features": int(len(feature_cols)),
                            **ticker_metrics,
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
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    metrics_path = EXPERIMENT_DIR / "baseline_metrics.csv"
    predictions_path = EXPERIMENT_DIR / "baseline_predictions.csv"
    features_path = EXPERIMENT_DIR / "feature_sets.json"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    features_path.write_text(json.dumps(feature_sets, indent=2, ensure_ascii=True), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "test_start": test_start.date().isoformat(),
                "final_holdout_months": FINAL_HOLDOUT_MONTHS,
                "models": list(MODEL_DEFS.keys()),
                "feature_sets": feature_sets,
                "alt_alignment": "calendar-day alternative data aggregated to trading session and used without extra one-day shift",
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    return {
        "dataset_path": dataset_path,
        "test_start": test_start,
        "feature_panel": feature_panel,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "features_path": features_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    result = run_baseline_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Predictions: {result['predictions_path']}")
    print(result["metrics"].query("split == 'test' and ticker == 'ALL'").to_string(index=False))


if __name__ == "__main__":
    main()
