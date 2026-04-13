from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from price_alt_baseline_experiment import (
    RANDOM_STATE,
    build_feature_panel,
    classification_metrics,
    engineer_ticker_features,
    get_feature_sets,
    locate_panel_dataset,
    rolling_zscore,
    safe_pct_change,
    shifted_series,
    signed_log1p,
    split_train_test,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "market_adjusted_surprise"
DATA_DIR = NOTEBOOK_DIR.parent / "data" / "equity_data"

TARGET_TICKERS = ["TSLA", "AAPL"]
TARGET_MODES = {
    "raw_direction": "target_raw_direction",
    "excess_direction": "target_excess_direction",
}

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
}

SURPRISE_ALT_FEATURES = [
    "trends_surprise_z20",
    "trends_jump_surprise_z20",
    "trends_spike_flag",
    "gdelt_articles_surprise_z20",
    "gdelt_articles_jump_surprise_z20",
    "gdelt_articles_spike_flag",
    "gdelt_sentiment_surprise_20",
    "gdelt_sentiment_vol_20",
    "subm_posts_surprise_z20",
    "subm_posts_spike_flag",
    "subm_score_surprise_z20",
    "subm_comments_surprise_z20",
    "subm_sentiment_mean_surprise_20",
    "subm_sentiment_gap",
    "comm_posts_surprise_z20",
    "comm_posts_spike_flag",
    "comm_score_surprise_z20",
    "comm_comments_surprise_z20",
    "comm_sentiment_mean_surprise_20",
    "comm_sentiment_gap",
]


def locate_market_benchmark() -> Path | None:
    candidates = [
        DATA_DIR / "stock-prices-data_spy.csv",
        DATA_DIR / "stock-prices-data_qqq.csv",
        DATA_DIR / "stock-prices-data_SPY.csv",
        DATA_DIR / "stock-prices-data_QQQ.csv",
        DATA_DIR / "spy_prices.csv",
        DATA_DIR / "qqq_prices.csv",
        OUTPUT_DIR / "datasets" / "stock-prices-data_spy.csv",
        OUTPUT_DIR / "datasets" / "stock-prices-data_qqq.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_panel(dataset_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in panel.columns:
        if col not in {"date", "ticker", "company_name", "y"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel


def load_market_benchmark(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    benchmark_path = locate_market_benchmark()
    if benchmark_path is not None:
        benchmark = pd.read_csv(benchmark_path)
        date_col = "Date" if "Date" in benchmark.columns else "date"
        price_candidates = ["stock_price", "Adj Close", "adj_close", "Close", "close"]
        price_col = next((col for col in price_candidates if col in benchmark.columns), None)
        if price_col is None:
            raise ValueError(f"Benchmark file {benchmark_path} has no supported price column.")

        benchmark["date"] = pd.to_datetime(benchmark[date_col], errors="coerce")
        benchmark["benchmark_price"] = pd.to_numeric(benchmark[price_col], errors="coerce")
        benchmark = benchmark.dropna(subset=["date", "benchmark_price"]).sort_values("date").reset_index(drop=True)
        benchmark["benchmark_future_return_1d"] = (
            benchmark["benchmark_price"].shift(-1) / benchmark["benchmark_price"] - 1
        )
        benchmark = benchmark[["date", "benchmark_future_return_1d"]].copy()
        metadata = {
            "benchmark_source": "external_market_file",
            "benchmark_path": str(benchmark_path),
        }
        return benchmark, metadata

    proxy = (
        panel.groupby("date", as_index=False)["future_return_1d"]
        .mean()
        .rename(columns={"future_return_1d": "benchmark_future_return_1d"})
    )
    metadata = {
        "benchmark_source": "equal_weight_panel_proxy",
        "benchmark_path": "",
    }
    return proxy, metadata


def add_surprise_reddit_block(frame: pd.DataFrame, features: pd.DataFrame, prefix: str) -> None:
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

    posts = shifted_series(frame, posts_col, fill="zero")
    if posts_col in frame.columns:
        log_posts = np.log1p(posts.clip(lower=0))
        posts_surprise = rolling_zscore(log_posts, 20)
        features[f"{prefix}posts_surprise_z20"] = posts_surprise
        features[f"{prefix}posts_spike_flag"] = (posts_surprise > 1.5).astype(float)

    if score_col in frame.columns:
        score = shifted_series(frame, score_col, fill="zero")
        features[f"{prefix}score_surprise_z20"] = rolling_zscore(signed_log1p(score), 20)

    if comments_col in frame.columns:
        comments = shifted_series(frame, comments_col, fill="zero")
        comment_signal = np.log1p(comments.clip(lower=0))
        features[f"{prefix}comments_surprise_z20"] = rolling_zscore(comment_signal, 20)

    if vader_col in frame.columns and finbert_col in frame.columns:
        vader = shifted_series(frame, vader_col, fill="zero")
        finbert = shifted_series(frame, finbert_col, fill="zero")
        sentiment_mean = 0.5 * (vader + finbert)
        features[f"{prefix}sentiment_mean_surprise_20"] = (
            sentiment_mean - sentiment_mean.rolling(20, min_periods=10).mean()
        )
        features[f"{prefix}sentiment_gap"] = (vader - finbert).abs()
    elif vader_col in frame.columns:
        vader = shifted_series(frame, vader_col, fill="zero")
        features[f"{prefix}sentiment_mean_surprise_20"] = (
            vader - vader.rolling(20, min_periods=10).mean()
        )
    elif finbert_col in frame.columns:
        finbert = shifted_series(frame, finbert_col, fill="zero")
        features[f"{prefix}sentiment_mean_surprise_20"] = (
            finbert - finbert.rolling(20, min_periods=10).mean()
        )


def engineer_surprise_ticker_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_values("date").reset_index(drop=True).copy()
    base = engineer_ticker_features(work)
    price_cols = [col for col in get_feature_sets(base)["price_only"] if col != "ticker_code"]

    features = base[
        ["date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir", "ticker_code"] + price_cols
    ].copy()

    trends_level = shifted_series(work, "google_trends_score", fill="ffill")
    trends_jump = safe_pct_change(trends_level, 1).clip(-1.0, 1.0)
    trends_surprise = rolling_zscore(trends_level, 20)
    features["trends_surprise_z20"] = trends_surprise
    features["trends_jump_surprise_z20"] = rolling_zscore(trends_jump, 20)
    features["trends_spike_flag"] = (trends_surprise.abs() > 1.5).astype(float)

    gdelt_articles = shifted_series(work, "gdelt_articles", fill="ffill")
    articles_signal = np.log1p(gdelt_articles.clip(lower=0))
    articles_jump = safe_pct_change(gdelt_articles, 1).clip(-3.0, 3.0)
    articles_surprise = rolling_zscore(articles_signal, 20)
    features["gdelt_articles_surprise_z20"] = articles_surprise
    features["gdelt_articles_jump_surprise_z20"] = rolling_zscore(articles_jump, 20)
    features["gdelt_articles_spike_flag"] = (articles_surprise > 1.5).astype(float)

    gdelt_sentiment = shifted_series(work, "gdelt_sentiment_score", fill="ffill")
    features["gdelt_sentiment_surprise_20"] = (
        gdelt_sentiment - gdelt_sentiment.rolling(20, min_periods=10).mean()
    )
    features["gdelt_sentiment_vol_20"] = gdelt_sentiment.rolling(20, min_periods=10).std()

    add_surprise_reddit_block(work, features, "subm_")
    add_surprise_reddit_block(work, features, "comm_")

    feature_cols = [
        col
        for col in features.columns
        if col not in ["date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir"]
    ]
    constant_cols = [col for col in feature_cols if features[col].nunique(dropna=True) <= 1]
    if constant_cols:
        features = features.drop(columns=constant_cols)

    return features


def build_surprise_feature_panel(panel: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, ticker_frame in panel.groupby("ticker", sort=True):
        pieces.append(engineer_surprise_ticker_features(ticker_frame))
    return pd.concat(pieces, ignore_index=True)


def add_target_modes(frame: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(benchmark, on="date", how="left")
    merged["target_raw_direction"] = np.where(
        merged["future_return_1d"].isna(),
        np.nan,
        (merged["future_return_1d"] > 0).astype(float),
    )
    merged["future_excess_return_1d"] = (
        merged["future_return_1d"] - merged["benchmark_future_return_1d"]
    )
    merged["target_excess_direction"] = np.where(
        merged["future_excess_return_1d"].isna(),
        np.nan,
        (merged["future_excess_return_1d"] > 0).astype(float),
    )
    return merged


def get_representation_feature_sets(
    current_frame: pd.DataFrame,
    surprise_frame: pd.DataFrame,
) -> dict[str, list[str]]:
    current_sets = get_feature_sets(current_frame)
    price_only = [col for col in current_sets["price_only"] if col != "ticker_code"]
    current_alt = [col for col in current_sets["price_plus_alt"] if col not in {"ticker_code", *price_only}]
    surprise_alt = [col for col in SURPRISE_ALT_FEATURES if col in surprise_frame.columns]

    return {
        "price_only": price_only,
        "price_plus_alt_current": list(dict.fromkeys(price_only + current_alt)),
        "price_plus_alt_surprise": list(dict.fromkeys(price_only + surprise_alt)),
    }


def fit_predict_frame(
    model_template,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_work = train_frame.dropna(subset=[target_col]).copy()
    test_work = test_frame.dropna(subset=[target_col]).copy()

    y_train = train_work[target_col].astype(int)
    if y_train.nunique() < 2:
        raise ValueError(f"Training target {target_col} has fewer than 2 classes.")

    model = clone(model_template)
    model.fit(train_work[feature_cols], y_train)

    outputs = []
    for split_name, split_frame in [("train", train_work), ("test", test_work)]:
        proba = model.predict_proba(split_frame[feature_cols])[:, 1]
        pred = (proba >= 0.5).astype(int)
        outputs.append(
            pd.DataFrame(
                {
                    "date": split_frame["date"].values,
                    "ticker": split_frame["ticker"].values,
                    "split": split_name,
                    "future_return_1d": split_frame["future_return_1d"].values,
                    "future_excess_return_1d": split_frame["future_excess_return_1d"].values,
                    "y_true": split_frame[target_col].astype(int).values,
                    "y_pred": pred,
                    "p_up": proba,
                }
            )
        )

    return outputs[0], outputs[1]


def metrics_from_predictions(
    prediction_frame: pd.DataFrame,
    model_name: str,
    feature_representation: str,
    target_mode: str,
    n_features: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name, split_frame in prediction_frame.groupby("split", sort=False):
        metrics_all = classification_metrics(
            split_frame["y_true"],
            split_frame["y_pred"].to_numpy(),
            split_frame["p_up"].to_numpy(),
        )
        rows.append(
            {
                "ticker": "ALL",
                "model_name": model_name,
                "feature_representation": feature_representation,
                "target_mode": target_mode,
                "split": split_name,
                "n_rows": int(len(split_frame)),
                "n_features": int(n_features),
                **metrics_all,
            }
        )

        for ticker, ticker_frame in split_frame.groupby("ticker", sort=False):
            metrics_ticker = classification_metrics(
                ticker_frame["y_true"],
                ticker_frame["y_pred"].to_numpy(),
                ticker_frame["p_up"].to_numpy(),
            )
            rows.append(
                {
                    "ticker": ticker,
                    "model_name": model_name,
                    "feature_representation": feature_representation,
                    "target_mode": target_mode,
                    "split": split_name,
                    "n_rows": int(len(ticker_frame)),
                    "n_features": int(n_features),
                    **metrics_ticker,
                }
            )
    return rows


def run_market_adjusted_surprise_experiment(
    dataset_path: Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    tickers = tickers or list(TARGET_TICKERS)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(dataset_path)
    benchmark, benchmark_metadata = load_market_benchmark(panel)

    current_frame = add_target_modes(build_feature_panel(panel), benchmark)
    surprise_frame = add_target_modes(build_surprise_feature_panel(panel), benchmark)
    feature_sets = get_representation_feature_sets(current_frame, surprise_frame)

    representation_frames = {
        "price_only": current_frame,
        "price_plus_alt_current": current_frame,
        "price_plus_alt_surprise": surprise_frame,
    }

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_representation, feature_cols in feature_sets.items():
        frame = representation_frames[feature_representation]
        frame = frame[frame["ticker"].isin(tickers)].copy()
        train_df, test_df, test_start = split_train_test(frame)

        for target_mode, target_col in TARGET_MODES.items():
            for model_name, model_template in MODEL_DEFS.items():
                predictions_per_ticker = []
                for ticker in tickers:
                    train_ticker = train_df[train_df["ticker"] == ticker].copy()
                    test_ticker = test_df[test_df["ticker"] == ticker].copy()
                    train_pred, test_pred = fit_predict_frame(
                        model_template=model_template,
                        train_frame=train_ticker,
                        test_frame=test_ticker,
                        feature_cols=feature_cols,
                        target_col=target_col,
                    )
                    predictions_per_ticker.extend([train_pred, test_pred])

                predictions = pd.concat(predictions_per_ticker, ignore_index=True)
                predictions["model_name"] = model_name
                predictions["feature_representation"] = feature_representation
                predictions["target_mode"] = target_mode
                prediction_frames.append(predictions)
                metrics_rows.extend(
                    metrics_from_predictions(
                        prediction_frame=predictions,
                        model_name=model_name,
                        feature_representation=feature_representation,
                        target_mode=target_mode,
                        n_features=len(feature_cols),
                    )
                )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["target_mode", "split", "ticker", "model_name", "feature_representation"]
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["target_mode", "feature_representation", "model_name", "split", "ticker", "date"]
    ).reset_index(drop=True)

    metrics_path = EXPERIMENT_DIR / "metrics.csv"
    predictions_path = EXPERIMENT_DIR / "predictions.csv"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"
    feature_sets_path = EXPERIMENT_DIR / "feature_sets.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    feature_sets_path.write_text(
        json.dumps(feature_sets, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "test_start": test_start.date().isoformat(),
                "tickers": tickers,
                "target_modes": list(TARGET_MODES.keys()),
                "models": list(MODEL_DEFS.keys()),
                "feature_sets": feature_sets,
                **benchmark_metadata,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    return {
        "dataset_path": dataset_path,
        "test_start": test_start,
        "current_frame": current_frame,
        "surprise_frame": surprise_frame,
        "benchmark": benchmark,
        "benchmark_metadata": benchmark_metadata,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "metadata_path": metadata_path,
        "feature_sets_path": feature_sets_path,
    }


def main() -> None:
    result = run_market_adjusted_surprise_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Predictions: {result['predictions_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(f"Benchmark source: {result['benchmark_metadata']['benchmark_source']}")
    print(
        result["metrics"]
        .query("split == 'test' and ticker == 'ALL'")
        .sort_values(["target_mode", "model_name", "feature_representation"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
