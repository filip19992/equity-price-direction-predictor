from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from price_alt_baseline_experiment import locate_panel_dataset, split_train_test
from strict_source_ablation_experiment import (
    TARGET_TICKERS,
    _LEGACY,
    build_source_ablation_panel,
    load_panel,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "final_shortlist_experiment"

SHORTLIST_FEATURES = {
    "AAPL": [
        "price_only",
        "price_plus_google_trends",
        "price_plus_reddit_sentiment",
    ],
    "TSLA": [
        "price_only",
        "price_plus_google_trends",
        "price_plus_reddit_attention",
    ],
}

SHORTLIST_MODELS = {
    name: _LEGACY.MODEL_DEFS[name]
    for name in ["logreg", "hist_gradient_boosting", "random_forest"]
}


def run_final_shortlist_experiment(
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
    feature_sets_by_ticker: dict[str, dict[str, list[str]]] = {}

    for ticker in tickers:
        ticker_frame = feature_panel[feature_panel["ticker"] == ticker].copy()
        ticker_feature_sets = {
            feature_set_name: feature_sets[feature_set_name]
            for feature_set_name in SHORTLIST_FEATURES.get(ticker, [])
            if feature_set_name in feature_sets
        }
        feature_sets_by_ticker[ticker] = ticker_feature_sets
        train_df, test_df, _ = split_train_test(ticker_frame)

        for feature_set_name, feature_cols in ticker_feature_sets.items():
            n_features = len(feature_cols)
            train_rows_per_feature = len(train_df.dropna(subset=["y_dir"])) / max(n_features, 1)

            for model_name, model_template in SHORTLIST_MODELS.items():
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
                            pd.NA
                            if pd.isna(tuning_info["validation_balanced_accuracy"])
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
                                pd.NA
                                if pd.isna(fold_tuning["validation_balanced_accuracy"])
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
        for ticker in tickers:
            for feature_set_name in SHORTLIST_FEATURES.get(ticker, []):
                if feature_set_name == "price_only":
                    continue
                alt_fold_df = cv_fold_df[
                    (cv_fold_df["ticker"] == ticker) & (cv_fold_df["feature_set"] == feature_set_name)
                ].copy()
                price_ticker_df = price_fold_df[price_fold_df["ticker"] == ticker].copy()
                merged = alt_fold_df.merge(
                    price_ticker_df,
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
        for model_name in SHORTLIST_MODELS:
            price_pred = ticker_test[
                (ticker_test["model_name"] == model_name) & (ticker_test["feature_set"] == "price_only")
            ].copy()
            price_pred = price_pred.sort_values("date")
            if price_pred.empty:
                continue
            for feature_set_name in SHORTLIST_FEATURES.get(ticker, []):
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
                delong_rows.append(
                    {
                        "ticker": ticker,
                        "model_name": model_name,
                        "feature_set_alt": feature_set_name,
                        **_LEGACY.delong_auc_pvalue(
                            merged["y_true"].to_numpy(),
                            merged["p_up_price"].to_numpy(),
                            merged["p_up_alt"].to_numpy(),
                        ),
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
    feature_sets_path = EXPERIMENT_DIR / "feature_sets_by_ticker.json"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    tuning_df.to_csv(tuning_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    cv_fold_df.to_csv(cv_fold_path, index=False)
    cv_delta_df.to_csv(cv_delta_path, index=False)
    delong_df.to_csv(delong_path, index=False)
    feature_sets_path.write_text(
        json.dumps(feature_sets_by_ticker, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "tickers": tickers,
                "models": list(SHORTLIST_MODELS.keys()),
                "feature_sets_by_ticker": feature_sets_by_ticker,
                "selection_rationale": {
                    "AAPL": ["price_plus_google_trends", "price_plus_reddit_sentiment"],
                    "TSLA": ["price_plus_google_trends", "price_plus_reddit_attention"],
                },
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
        "feature_sets_by_ticker": feature_sets_by_ticker,
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
        "feature_sets_path": feature_sets_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    result = run_final_shortlist_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"DeLong summary: {result['delong_path']}")
    print(
        result["metrics"]
        .query("split == 'test'")
        .sort_values(["ticker", "model_name", "feature_set"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
