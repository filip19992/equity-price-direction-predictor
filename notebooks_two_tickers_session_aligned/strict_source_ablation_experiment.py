from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from price_alt_baseline_experiment import (
    engineer_ticker_features,
    get_feature_sets,
    locate_panel_dataset,
    rolling_zscore,
    signed_log1p,
    split_train_test,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "strict_source_ablation"
TARGET_TICKERS = ["TSLA", "AAPL"]

_LEGACY_PATH = NOTEBOOK_DIR.parent / "notebooks_two_tickers" / "strict_source_ablation_experiment.py"
_LEGACY_BASELINE_PATH = NOTEBOOK_DIR.parent / "notebooks_two_tickers" / "price_alt_baseline_experiment.py"

_CURRENT_BASELINE_MODULE = sys.modules.get("price_alt_baseline_experiment")
_LEGACY_BASELINE_SPEC = importlib.util.spec_from_file_location(
    "price_alt_baseline_experiment",
    _LEGACY_BASELINE_PATH,
)
if _LEGACY_BASELINE_SPEC is None or _LEGACY_BASELINE_SPEC.loader is None:
    raise ImportError(f"Could not load legacy baseline helpers from {_LEGACY_BASELINE_PATH}")
_LEGACY_BASELINE = importlib.util.module_from_spec(_LEGACY_BASELINE_SPEC)
sys.modules["price_alt_baseline_experiment"] = _LEGACY_BASELINE
_LEGACY_BASELINE_SPEC.loader.exec_module(_LEGACY_BASELINE)

_LEGACY_SPEC = importlib.util.spec_from_file_location("_legacy_strict_source_ablation_experiment", _LEGACY_PATH)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:
    raise ImportError(f"Could not load legacy strict source ablation helpers from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_LEGACY_SPEC)
_LEGACY_SPEC.loader.exec_module(_LEGACY)
if _CURRENT_BASELINE_MODULE is not None:
    sys.modules["price_alt_baseline_experiment"] = _CURRENT_BASELINE_MODULE
else:
    sys.modules.pop("price_alt_baseline_experiment", None)


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


def engineer_source_ablation_ticker(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    work = frame.sort_values("date").reset_index(drop=True).copy()
    base = engineer_ticker_features(work)
    price_cols = [col for col in get_feature_sets(base)["price_only"] if col != "ticker_code"]

    metadata_cols = ["date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir"]
    features = base[metadata_cols + price_cols].copy()
    feature_sets = {"price_only": list(price_cols)}

    trends_level = current_series(work, "google_trends_score", fill="ffill")
    trends_cols = add_source_transforms(features, "trends", {"level": trends_level})
    feature_sets["price_plus_google_trends"] = list(dict.fromkeys(price_cols + trends_cols))

    gdelt_articles = np.log1p(current_series(work, "gdelt_articles", fill="ffill").clip(lower=0))
    gdelt_robust = current_series(work, "gdelt_robust", fill="ffill")
    gdelt_sentiment = current_series(work, "gdelt_sentiment_score", fill="ffill")
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

    subm_vader = current_series(work, "subm_reddit_vader_weighted_mean", fill="zero")
    comm_vader = current_series(work, "comm_reddit_vader_weighted_mean", fill="zero")
    subm_finbert = current_series(work, "subm_reddit_finbert_weighted_mean", fill="zero")
    comm_finbert = current_series(work, "comm_reddit_finbert_weighted_mean", fill="zero")
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

    subm_posts = current_series(work, "subm_reddit_posts", fill="zero")
    comm_posts = current_series(work, "comm_reddit_posts", fill="zero")
    subm_comments = current_series(work, "subm_reddit_comments_sum", fill="zero")
    comm_comments = current_series(work, "comm_reddit_comments_sum", fill="zero")
    subm_score = current_series(work, "subm_reddit_score_sum", fill="zero")
    comm_score = current_series(work, "comm_reddit_score_sum", fill="zero")
    subm_weight = current_series(work, "subm_reddit_weight_sum", fill="zero")
    comm_weight = current_series(work, "comm_reddit_weight_sum", fill="zero")
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
        train_df, test_df, _ = split_train_test(ticker_frame)

        for feature_set_name, feature_cols in feature_sets.items():
            n_features = len(feature_cols)
            train_rows_per_feature = len(train_df.dropna(subset=["y_dir"])) / max(n_features, 1)

            for model_name, model_template in _LEGACY.MODEL_DEFS.items():
                model, threshold, tuning_info = _LEGACY.fit_model_with_threshold(
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

                train_pred = _LEGACY.predict_frame(model, train_df, feature_cols, threshold, "train")
                test_pred = _LEGACY.predict_frame(model, test_df, feature_cols, threshold, "test")
                predictions = pd.concat([train_pred, test_pred], ignore_index=True)
                predictions["model_name"] = model_name
                predictions["feature_set"] = feature_set_name
                prediction_frames.append(predictions)

                metrics_rows.extend(
                    _LEGACY.metrics_from_predictions(
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

                bootstrap_summary = _LEGACY.bootstrap_metric_summary(test_pred)
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

                for fold_idx, fold_train, fold_val in _LEGACY.build_outer_folds(train_df):
                    fold_model, fold_threshold, fold_tuning = _LEGACY.fit_model_with_threshold(
                        model_name=model_name,
                        model_template=model_template,
                        train_frame=fold_train,
                        feature_cols=feature_cols,
                    )
                    fold_pred = _LEGACY.predict_frame(
                        fold_model,
                        fold_val,
                        feature_cols,
                        fold_threshold,
                        split_name="validation",
                    )
                    fold_metrics = _LEGACY.classification_metrics(
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
        for model_name in _LEGACY.MODEL_DEFS:
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
                delong_result = _LEGACY.delong_auc_pvalue(
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
                "models": list(_LEGACY.MODEL_DEFS.keys()),
                "feature_sets": feature_sets,
                "final_holdout_months": 6,
                "alt_time_alignment": "calendar-day alternative data aggregated to trading session and used as current-session features",
                "threshold_tuned_models": sorted(_LEGACY.THRESHOLD_TUNED_MODELS),
                "threshold_grid": _LEGACY.THRESHOLD_GRID.tolist(),
                "bootstrap_samples": _LEGACY.BOOTSTRAP_SAMPLES,
                "bootstrap_block_length": _LEGACY.BOOTSTRAP_BLOCK_LENGTH,
                "outer_folds": _LEGACY.OUTER_FOLDS,
                "outer_validation_days": _LEGACY.OUTER_VAL_DAYS,
                "inner_validation_fraction": _LEGACY.INNER_VAL_FRACTION,
                "note": "Alternative sources are evaluated separately on the session-aligned dataset.",
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
