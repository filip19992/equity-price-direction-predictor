from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from price_alt_baseline_experiment import (
    RANDOM_STATE,
    classification_metrics,
    engineer_ticker_features,
    get_feature_sets,
    locate_panel_dataset,
    rolling_zscore,
    shifted_series,
    signed_log1p,
    split_train_test,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "strict_source_ablation"

TARGET_TICKERS = ["TSLA", "AAPL"]
THRESHOLD_TUNED_MODELS = {"svm_rbf", "mlp_small"}
THRESHOLD_GRID = np.round(np.linspace(0.30, 0.70, 41), 3)
BOOTSTRAP_SAMPLES = 400
BOOTSTRAP_BLOCK_LENGTH = 5
OUTER_FOLDS = 4
OUTER_VAL_DAYS = 63
MIN_TRAIN_DAYS_FOR_FOLDS = 252
INNER_VAL_FRACTION = 0.20
INNER_MIN_VAL_DAYS = 40
INNER_MIN_TRAIN_DAYS = 120

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
    "svm_rbf": Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
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
    "mlp_small": Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    alpha=0.01,
                    batch_size=32,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}


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


def engineer_source_ablation_ticker(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    work = frame.sort_values("date").reset_index(drop=True).copy()
    base = engineer_ticker_features(work)
    price_cols = [col for col in get_feature_sets(base)["price_only"] if col != "ticker_code"]

    metadata_cols = ["date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir"]
    features = base[metadata_cols + price_cols].copy()

    feature_sets = {"price_only": list(price_cols)}

    trends_level = shifted_series(work, "google_trends_score", fill="ffill")
    trends_cols = add_source_transforms(features, "trends", {"level": trends_level})
    feature_sets["price_plus_google_trends"] = list(dict.fromkeys(price_cols + trends_cols))

    gdelt_articles = np.log1p(shifted_series(work, "gdelt_articles", fill="ffill").clip(lower=0))
    gdelt_robust = shifted_series(work, "gdelt_robust", fill="ffill")
    gdelt_sentiment = shifted_series(work, "gdelt_sentiment_score", fill="ffill")
    gdelt_cols = add_source_transforms(
        features,
        "gdelt",
        {
            "articles": gdelt_articles,
            "robust": gdelt_robust,
            "sentiment": gdelt_sentiment,
        },
    )
    feature_sets["price_plus_gdelt"] = list(dict.fromkeys(price_cols + gdelt_cols))

    subm_vader = shifted_series(work, "subm_reddit_vader_weighted_mean", fill="zero")
    comm_vader = shifted_series(work, "comm_reddit_vader_weighted_mean", fill="zero")
    subm_finbert = shifted_series(work, "subm_reddit_finbert_weighted_mean", fill="zero")
    comm_finbert = shifted_series(work, "comm_reddit_finbert_weighted_mean", fill="zero")
    reddit_vader = average_available([subm_vader, comm_vader])
    reddit_finbert = average_available([subm_finbert, comm_finbert])
    reddit_sentiment = average_available([reddit_vader, reddit_finbert])
    reddit_sent_gap = (reddit_vader - reddit_finbert).abs()
    reddit_sent_cols = add_source_transforms(
        features,
        "reddit_sent",
        {
            "vader": reddit_vader,
            "finbert": reddit_finbert,
            "direction": reddit_sentiment,
            "gap": reddit_sent_gap,
        },
    )
    feature_sets["price_plus_reddit_sentiment"] = list(dict.fromkeys(price_cols + reddit_sent_cols))

    subm_posts = shifted_series(work, "subm_reddit_posts", fill="zero")
    comm_posts = shifted_series(work, "comm_reddit_posts", fill="zero")
    subm_comments = shifted_series(work, "subm_reddit_comments_sum", fill="zero")
    comm_comments = shifted_series(work, "comm_reddit_comments_sum", fill="zero")
    subm_score = shifted_series(work, "subm_reddit_score_sum", fill="zero")
    comm_score = shifted_series(work, "comm_reddit_score_sum", fill="zero")
    subm_weight = shifted_series(work, "subm_reddit_weight_sum", fill="zero")
    comm_weight = shifted_series(work, "comm_reddit_weight_sum", fill="zero")
    reddit_attention_cols = add_source_transforms(
        features,
        "reddit_attn",
        {
            "posts_total": np.log1p((subm_posts + comm_posts).clip(lower=0)),
            "comments_total": np.log1p((subm_comments + comm_comments).clip(lower=0)),
            "score_total": signed_log1p(subm_score + comm_score),
            "weight_total": np.log1p((subm_weight + comm_weight).clip(lower=0)),
        },
    )
    feature_sets["price_plus_reddit_attention"] = list(dict.fromkeys(price_cols + reddit_attention_cols))

    return features, feature_sets


def build_source_ablation_panel(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    pieces = []
    feature_sets_agg: dict[str, list[str]] = {}

    for _, ticker_frame in panel.groupby("ticker", sort=True):
        ticker_features, ticker_feature_sets = engineer_source_ablation_ticker(ticker_frame)
        pieces.append(ticker_features)
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

    return features, feature_sets_agg


def split_inner_train_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    unique_dates = pd.Series(frame["date"].dropna().sort_values().unique())
    n_dates = len(unique_dates)
    if n_dates < (INNER_MIN_TRAIN_DAYS + INNER_MIN_VAL_DAYS):
        return None

    val_days = max(INNER_MIN_VAL_DAYS, int(round(n_dates * INNER_VAL_FRACTION)))
    val_days = min(val_days, max(INNER_MIN_VAL_DAYS, n_dates // 3))
    cutoff_idx = n_dates - val_days
    if cutoff_idx < INNER_MIN_TRAIN_DAYS:
        return None

    cutoff_date = unique_dates.iloc[cutoff_idx]
    inner_train = frame[frame["date"] < cutoff_date].copy()
    inner_val = frame[frame["date"] >= cutoff_date].copy()
    if inner_train.empty or inner_val.empty:
        return None
    return inner_train, inner_val


def select_decision_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_score = -np.inf
    for threshold in THRESHOLD_GRID:
        pred = (scores >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def fit_model_with_threshold(
    model_name: str,
    model_template,
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_dir",
) -> tuple[object, float, dict[str, object]]:
    train_work = train_frame.dropna(subset=[target_col]).copy()
    y_train = train_work[target_col].astype(int)
    if y_train.nunique() < 2:
        raise ValueError(f"Training target {target_col} has fewer than 2 classes.")

    threshold = 0.50
    tuning_info: dict[str, object] = {
        "threshold_tuned": model_name in THRESHOLD_TUNED_MODELS,
        "validation_rows": 0,
        "validation_balanced_accuracy": np.nan,
    }

    if model_name in THRESHOLD_TUNED_MODELS:
        inner_split = split_inner_train_validation(train_work)
        if inner_split is not None:
            inner_train, inner_val = inner_split
            inner_y_train = inner_train[target_col].astype(int)
            inner_y_val = inner_val[target_col].astype(int)
            if inner_y_train.nunique() >= 2 and inner_y_val.nunique() >= 2:
                inner_model = clone(model_template)
                inner_model.fit(inner_train[feature_cols], inner_y_train)
                val_scores = inner_model.predict_proba(inner_val[feature_cols])[:, 1]
                threshold, val_bal_acc = select_decision_threshold(
                    inner_y_val,
                    val_scores,
                )
                tuning_info["validation_rows"] = int(len(inner_val))
                tuning_info["validation_balanced_accuracy"] = float(val_bal_acc)

    final_model = clone(model_template)
    final_model.fit(train_work[feature_cols], y_train)
    return final_model, threshold, tuning_info


def predict_frame(
    model,
    frame: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
    split_name: str,
    target_col: str = "y_dir",
) -> pd.DataFrame:
    work = frame.dropna(subset=[target_col]).copy()
    proba = model.predict_proba(work[feature_cols])[:, 1]
    pred = (proba >= threshold).astype(int)
    return pd.DataFrame(
        {
            "date": work["date"].values,
            "ticker": work["ticker"].values,
            "split": split_name,
            "future_return_1d": work["future_return_1d"].values,
            "y_true": work[target_col].astype(int).values,
            "y_pred": pred,
            "p_up": proba,
            "decision_threshold": threshold,
        }
    )


def metrics_from_predictions(
    prediction_frame: pd.DataFrame,
    ticker: str,
    model_name: str,
    feature_set_name: str,
    n_features: int,
    train_rows_per_feature: float,
    validation_rows: int,
    threshold_tuned: bool,
    validation_balanced_accuracy: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    threshold = float(prediction_frame["decision_threshold"].iloc[0])
    for split_name, split_frame in prediction_frame.groupby("split", sort=False):
        metrics = classification_metrics(
            split_frame["y_true"],
            split_frame["y_pred"].to_numpy(),
            split_frame["p_up"].to_numpy(),
        )
        rows.append(
            {
                "ticker": ticker,
                "model_name": model_name,
                "feature_set": feature_set_name,
                "split": split_name,
                "n_rows": int(len(split_frame)),
                "n_features": int(n_features),
                "train_rows_per_feature": float(train_rows_per_feature),
                "decision_threshold": threshold,
                "threshold_tuned": bool(threshold_tuned),
                "validation_rows": int(validation_rows),
                "validation_balanced_accuracy": (
                    np.nan if np.isnan(validation_balanced_accuracy) else float(validation_balanced_accuracy)
                ),
                **metrics,
            }
        )
    return rows


def build_outer_folds(frame: pd.DataFrame) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    unique_dates = pd.Series(frame["date"].dropna().sort_values().unique())
    n_dates = len(unique_dates)
    if n_dates <= MIN_TRAIN_DAYS_FOR_FOLDS:
        return []

    val_days = min(OUTER_VAL_DAYS, max(20, n_dates // (OUTER_FOLDS + 2)))
    max_folds = max(0, (n_dates - MIN_TRAIN_DAYS_FOR_FOLDS) // val_days)
    n_folds = min(OUTER_FOLDS, max_folds)
    if n_folds == 0:
        return []

    start_idx = n_dates - n_folds * val_days
    folds: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for fold_idx in range(n_folds):
        val_start = start_idx + fold_idx * val_days
        val_end = min(n_dates, val_start + val_days)
        train_dates = unique_dates.iloc[:val_start]
        val_dates = unique_dates.iloc[val_start:val_end]
        train_fold = frame[frame["date"].isin(train_dates)].copy()
        val_fold = frame[frame["date"].isin(val_dates)].copy()
        if train_fold.empty or val_fold.empty:
            continue
        folds.append((fold_idx + 1, train_fold, val_fold))
    return folds


def moving_block_bootstrap_indices(
    n_obs: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_obs <= block_length:
        return rng.integers(0, n_obs, size=n_obs)

    n_blocks = int(math.ceil(n_obs / block_length))
    starts = rng.integers(0, n_obs - block_length + 1, size=n_blocks)
    indices = np.concatenate([np.arange(start, start + block_length) for start in starts])[:n_obs]
    return indices


def bootstrap_metric_summary(
    prediction_frame: pd.DataFrame,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
) -> dict[str, float]:
    y_true = prediction_frame["y_true"].to_numpy()
    y_score = prediction_frame["p_up"].to_numpy()
    threshold = float(prediction_frame["decision_threshold"].iloc[0])
    rng = np.random.default_rng(RANDOM_STATE)

    bal_acc_samples: list[float] = []
    roc_auc_samples: list[float] = []

    for _ in range(n_bootstrap):
        idx = moving_block_bootstrap_indices(len(prediction_frame), block_length, rng)
        y_true_boot = y_true[idx]
        y_score_boot = y_score[idx]
        y_pred_boot = (y_score_boot >= threshold).astype(int)
        bal_acc_samples.append(float(balanced_accuracy_score(y_true_boot, y_pred_boot)))
        if len(np.unique(y_true_boot)) > 1:
            roc_auc_samples.append(
                float(classification_metrics(pd.Series(y_true_boot), y_pred_boot, y_score_boot)["roc_auc"])
            )

    bal_arr = np.asarray(bal_acc_samples, dtype="float64")
    auc_arr = np.asarray(roc_auc_samples, dtype="float64") if roc_auc_samples else np.asarray([], dtype="float64")

    return {
        "balanced_accuracy_ci_low": float(np.nanpercentile(bal_arr, 2.5)),
        "balanced_accuracy_ci_high": float(np.nanpercentile(bal_arr, 97.5)),
        "roc_auc_ci_low": float(np.nanpercentile(auc_arr, 2.5)) if auc_arr.size else np.nan,
        "roc_auc_ci_high": float(np.nanpercentile(auc_arr, 97.5)) if auc_arr.size else np.nan,
    }


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    midranks = np.zeros(len(values), dtype="float64")

    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    result = np.empty(len(values), dtype="float64")
    result[order] = midranks
    return result


def fast_delong(predictions_sorted: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = positive_count
    n = predictions_sorted.shape[1] - m
    k = predictions_sorted.shape[0]

    positive_examples = predictions_sorted[:, :m]
    negative_examples = predictions_sorted[:, m:]
    tx = np.empty((k, m), dtype="float64")
    ty = np.empty((k, n), dtype="float64")
    tz = np.empty((k, m + n), dtype="float64")

    for row in range(k):
        tx[row] = compute_midrank(positive_examples[row])
        ty[row] = compute_midrank(negative_examples[row])
        tz[row] = compute_midrank(predictions_sorted[row])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if np.ndim(sx) == 0:
        sx = np.asarray([[sx]], dtype="float64")
    if np.ndim(sy) == 0:
        sy = np.asarray([[sy]], dtype="float64")
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_auc_pvalue(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {
            "auc_a": np.nan,
            "auc_b": np.nan,
            "auc_diff": np.nan,
            "auc_diff_se": np.nan,
            "z_stat": np.nan,
            "p_value": np.nan,
        }

    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    preds_sorted = np.vstack([score_a[order], score_b[order]])
    positive_count = int(y_true_sorted.sum())
    aucs, covariance = fast_delong(preds_sorted, positive_count)

    diff = float(aucs[0] - aucs[1])
    var = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    if var <= 0:
        return {
            "auc_a": float(aucs[0]),
            "auc_b": float(aucs[1]),
            "auc_diff": diff,
            "auc_diff_se": np.nan,
            "z_stat": np.nan,
            "p_value": np.nan,
        }

    se = math.sqrt(var)
    z_stat = diff / se
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0))))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": diff,
        "auc_diff_se": float(se),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
    }


def load_panel(dataset_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in panel.columns:
        if col not in {"date", "ticker", "company_name", "y"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel


def run_strict_source_ablation_experiment(
    dataset_path: Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    tickers = tickers or list(TARGET_TICKERS)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(dataset_path)
    panel = panel[panel["ticker"].isin(tickers)].copy()
    feature_panel, feature_sets = build_source_ablation_panel(panel)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    cv_fold_rows: list[dict[str, object]] = []
    delong_rows: list[dict[str, object]] = []

    for ticker in tickers:
        ticker_frame = feature_panel[feature_panel["ticker"] == ticker].copy()
        train_df, test_df, test_start = split_train_test(ticker_frame)

        for feature_set_name, feature_cols in feature_sets.items():
            n_features = len(feature_cols)
            train_rows_per_feature = len(train_df.dropna(subset=["y_dir"])) / max(n_features, 1)

            for model_name, model_template in MODEL_DEFS.items():
                model, threshold, tuning_info = fit_model_with_threshold(
                    model_name=model_name,
                    model_template=model_template,
                    train_frame=train_df,
                    feature_cols=feature_cols,
                )

                tuning_rows.append(
                    {
                        "ticker": ticker,
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "split_scope": "final_test",
                        "decision_threshold": float(threshold),
                        "threshold_tuned": bool(tuning_info["threshold_tuned"]),
                        "validation_rows": int(tuning_info["validation_rows"]),
                        "validation_balanced_accuracy": (
                            np.nan
                            if np.isnan(tuning_info["validation_balanced_accuracy"])
                            else float(tuning_info["validation_balanced_accuracy"])
                        ),
                    }
                )

                train_pred = predict_frame(model, train_df, feature_cols, threshold, "train")
                test_pred = predict_frame(model, test_df, feature_cols, threshold, "test")
                predictions = pd.concat([train_pred, test_pred], ignore_index=True)
                predictions["model_name"] = model_name
                predictions["feature_set"] = feature_set_name
                prediction_frames.append(predictions)

                metrics_rows.extend(
                    metrics_from_predictions(
                        prediction_frame=predictions,
                        ticker=ticker,
                        model_name=model_name,
                        feature_set_name=feature_set_name,
                        n_features=n_features,
                        train_rows_per_feature=train_rows_per_feature,
                        validation_rows=int(tuning_info["validation_rows"]),
                        threshold_tuned=bool(tuning_info["threshold_tuned"]),
                        validation_balanced_accuracy=tuning_info["validation_balanced_accuracy"],
                    )
                )

                bootstrap_summary = bootstrap_metric_summary(test_pred)
                bootstrap_rows.append(
                    {
                        "ticker": ticker,
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "split": "test",
                        "decision_threshold": float(threshold),
                        **bootstrap_summary,
                    }
                )

                for fold_idx, fold_train, fold_val in build_outer_folds(train_df):
                    fold_model, fold_threshold, fold_tuning = fit_model_with_threshold(
                        model_name=model_name,
                        model_template=model_template,
                        train_frame=fold_train,
                        feature_cols=feature_cols,
                    )
                    fold_pred = predict_frame(
                        fold_model,
                        fold_val,
                        feature_cols,
                        fold_threshold,
                        split_name="validation",
                    )
                    fold_metrics = classification_metrics(
                        fold_pred["y_true"],
                        fold_pred["y_pred"].to_numpy(),
                        fold_pred["p_up"].to_numpy(),
                    )
                    cv_fold_rows.append(
                        {
                            "ticker": ticker,
                            "fold": int(fold_idx),
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "n_train_rows": int(len(fold_train)),
                            "n_validation_rows": int(len(fold_val)),
                            "decision_threshold": float(fold_threshold),
                            "threshold_tuned": bool(fold_tuning["threshold_tuned"]),
                            "validation_rows_inner": int(fold_tuning["validation_rows"]),
                            "validation_balanced_accuracy_inner": (
                                np.nan
                                if np.isnan(fold_tuning["validation_balanced_accuracy"])
                                else float(fold_tuning["validation_balanced_accuracy"])
                            ),
                            **fold_metrics,
                        }
                    )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["ticker", "split", "model_name", "feature_set"]
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["ticker", "model_name", "feature_set", "split", "date"]
    ).reset_index(drop=True)
    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["ticker", "model_name", "feature_set", "split_scope"]
    ).reset_index(drop=True)
    bootstrap_df = pd.DataFrame(bootstrap_rows).sort_values(
        ["ticker", "model_name", "feature_set"]
    ).reset_index(drop=True)
    cv_fold_df = pd.DataFrame(cv_fold_rows).sort_values(
        ["ticker", "fold", "model_name", "feature_set"]
    ).reset_index(drop=True)

    cv_delta_rows: list[dict[str, object]] = []
    if not cv_fold_df.empty:
        price_fold_df = cv_fold_df[cv_fold_df["feature_set"] == "price_only"].copy()
        for feature_set_name in feature_sets:
            if feature_set_name == "price_only":
                continue
            alt_fold_df = cv_fold_df[cv_fold_df["feature_set"] == feature_set_name].copy()
            merged = alt_fold_df.merge(
                price_fold_df,
                on=["ticker", "fold", "model_name"],
                suffixes=("_alt", "_price"),
            )
            for _, row in merged.iterrows():
                cv_delta_rows.append(
                    {
                        "ticker": row["ticker"],
                        "fold": int(row["fold"]),
                        "model_name": row["model_name"],
                        "feature_set_alt": feature_set_name,
                        "balanced_accuracy_alt": float(row["balanced_accuracy_alt"]),
                        "balanced_accuracy_price": float(row["balanced_accuracy_price"]),
                        "delta_balanced_accuracy": float(
                            row["balanced_accuracy_alt"] - row["balanced_accuracy_price"]
                        ),
                        "roc_auc_alt": float(row["roc_auc_alt"]),
                        "roc_auc_price": float(row["roc_auc_price"]),
                        "delta_roc_auc": float(row["roc_auc_alt"] - row["roc_auc_price"]),
                    }
                )
    cv_delta_df = pd.DataFrame(cv_delta_rows)
    if not cv_delta_df.empty:
        cv_delta_df = cv_delta_df.sort_values(
            ["ticker", "fold", "model_name", "feature_set_alt"]
        ).reset_index(drop=True)

    test_predictions = predictions_df[predictions_df["split"] == "test"].copy()
    for ticker in tickers:
        ticker_test = test_predictions[test_predictions["ticker"] == ticker].copy()
        for model_name in MODEL_DEFS:
            price_pred = ticker_test[
                (ticker_test["model_name"] == model_name) & (ticker_test["feature_set"] == "price_only")
            ].copy()
            price_pred = price_pred.sort_values("date")
            if price_pred.empty:
                continue
            for feature_set_name in feature_sets:
                if feature_set_name == "price_only":
                    continue
                alt_pred = ticker_test[
                    (ticker_test["model_name"] == model_name) & (ticker_test["feature_set"] == feature_set_name)
                ].copy()
                alt_pred = alt_pred.sort_values("date")
                if alt_pred.empty:
                    continue
                merged = price_pred.merge(
                    alt_pred,
                    on=["date", "ticker", "split", "y_true"],
                    suffixes=("_price", "_alt"),
                )
                delong_result = delong_auc_pvalue(
                    merged["y_true"].to_numpy(),
                    merged["p_up_price"].to_numpy(),
                    merged["p_up_alt"].to_numpy(),
                )
                delong_rows.append(
                    {
                        "ticker": ticker,
                        "model_name": model_name,
                        "feature_set_alt": feature_set_name,
                        **delong_result,
                    }
                )

    delong_df = pd.DataFrame(delong_rows)
    if not delong_df.empty:
        delong_df = delong_df.sort_values(
            ["ticker", "model_name", "feature_set_alt"]
        ).reset_index(drop=True)

    metrics_path = EXPERIMENT_DIR / "metrics.csv"
    predictions_path = EXPERIMENT_DIR / "predictions.csv"
    tuning_path = EXPERIMENT_DIR / "threshold_tuning.csv"
    bootstrap_path = EXPERIMENT_DIR / "bootstrap_summary.csv"
    cv_fold_path = EXPERIMENT_DIR / "cv_fold_metrics.csv"
    cv_delta_path = EXPERIMENT_DIR / "cv_fold_deltas_vs_price_only.csv"
    delong_path = EXPERIMENT_DIR / "delong_vs_price_only.csv"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"
    feature_sets_path = EXPERIMENT_DIR / "feature_sets.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    tuning_df.to_csv(tuning_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    cv_fold_df.to_csv(cv_fold_path, index=False)
    cv_delta_df.to_csv(cv_delta_path, index=False)
    delong_df.to_csv(delong_path, index=False)
    feature_sets_path.write_text(json.dumps(feature_sets, indent=2, ensure_ascii=True), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "tickers": tickers,
                "models": list(MODEL_DEFS.keys()),
                "feature_sets": feature_sets,
                "final_holdout_months": 6,
                "alt_time_alignment": "all alternative data shifted by one trading day before lag/rolling transforms",
                "threshold_tuned_models": sorted(THRESHOLD_TUNED_MODELS),
                "threshold_grid": THRESHOLD_GRID.tolist(),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
                "outer_folds": OUTER_FOLDS,
                "outer_validation_days": OUTER_VAL_DAYS,
                "inner_validation_fraction": INNER_VAL_FRACTION,
                "note": "Alternative sources are evaluated separately to limit feature count relative to sample size.",
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
        "threshold_tuning": tuning_df,
        "bootstrap_summary": bootstrap_df,
        "cv_fold_metrics": cv_fold_df,
        "cv_fold_deltas": cv_delta_df,
        "delong_summary": delong_df,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "tuning_path": tuning_path,
        "bootstrap_path": bootstrap_path,
        "cv_fold_path": cv_fold_path,
        "cv_delta_path": cv_delta_path,
        "delong_path": delong_path,
        "metadata_path": metadata_path,
        "feature_sets_path": feature_sets_path,
    }


def main() -> None:
    result = run_strict_source_ablation_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Predictions: {result['predictions_path']}")
    print(f"Threshold tuning: {result['tuning_path']}")
    print(f"Bootstrap summary: {result['bootstrap_path']}")
    print(f"Fold metrics: {result['cv_fold_path']}")
    print(f"DeLong summary: {result['delong_path']}")
    print(
        result["metrics"]
        .query("split == 'test'")
        .sort_values(["ticker", "model_name", "feature_set"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
