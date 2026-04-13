from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alt_transform_screening_experiment import (
    EXPERIMENT_DIR as SCREENING_DIR,
    TARGET_TICKERS,
    build_candidate_feature_panel,
    run_alt_transform_screening_experiment,
)
from price_alt_baseline_experiment import classification_metrics, locate_panel_dataset, split_train_test
from strict_source_ablation_experiment import (
    MODEL_DEFS,
    bootstrap_metric_summary,
    build_outer_folds,
    delong_auc_pvalue,
    fit_model_with_threshold,
    load_panel,
    metrics_from_predictions,
    predict_frame,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "selected_alt_final_models"


def load_or_create_selection_manifest(selection_path: Path | None = None) -> dict[str, object]:
    selection_path = selection_path or (SCREENING_DIR / "selected_feature_manifest.json")
    if not selection_path.exists():
        run_alt_transform_screening_experiment()
    return json.loads(selection_path.read_text(encoding="utf-8"))


def build_feature_sets_for_ticker(
    ticker: str,
    ticker_frame: pd.DataFrame,
    selection_manifest: dict[str, object],
) -> dict[str, list[str]]:
    ticker_info = selection_manifest["tickers"].get(ticker, {})
    price_cols = [col for col in ticker_info.get("price_only_features", []) if col in ticker_frame.columns]
    selected_alt = [
        col for col in ticker_info.get("selected_alt_features", []) if col in ticker_frame.columns
    ]
    return {
        "price_only": price_cols,
        "price_plus_selected_alt": list(dict.fromkeys(price_cols + selected_alt)),
    }


def run_selected_alt_final_models_experiment(
    dataset_path: Path | None = None,
    selection_path: Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    tickers = tickers or list(TARGET_TICKERS)
    selection_manifest = load_or_create_selection_manifest(selection_path)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(dataset_path)
    panel = panel[panel["ticker"].isin(tickers)].copy()
    feature_panel, _, _ = build_candidate_feature_panel(panel)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    cv_fold_rows: list[dict[str, object]] = []
    delong_rows: list[dict[str, object]] = []
    feature_sets_by_ticker: dict[str, dict[str, list[str]]] = {}

    for ticker in tickers:
        ticker_frame = feature_panel[feature_panel["ticker"] == ticker].copy()
        feature_sets = build_feature_sets_for_ticker(ticker, ticker_frame, selection_manifest)
        feature_sets_by_ticker[ticker] = feature_sets
        train_df, test_df, _ = split_train_test(ticker_frame)

        for feature_set_name, feature_cols in feature_sets.items():
            if not feature_cols:
                continue
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
        alt_fold_df = cv_fold_df[cv_fold_df["feature_set"] == "price_plus_selected_alt"].copy()
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
                    "feature_set_alt": "price_plus_selected_alt",
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
            ["ticker", "fold", "model_name"]
        ).reset_index(drop=True)

    test_predictions = predictions_df[predictions_df["split"] == "test"].copy()
    for ticker in tickers:
        ticker_test = test_predictions[test_predictions["ticker"] == ticker].copy()
        for model_name in MODEL_DEFS:
            price_pred = ticker_test[
                (ticker_test["model_name"] == model_name)
                & (ticker_test["feature_set"] == "price_only")
            ].copy()
            alt_pred = ticker_test[
                (ticker_test["model_name"] == model_name)
                & (ticker_test["feature_set"] == "price_plus_selected_alt")
            ].copy()
            if price_pred.empty or alt_pred.empty:
                continue
            merged = price_pred.sort_values("date").merge(
                alt_pred.sort_values("date"),
                on=["date", "ticker", "split", "y_true"],
                suffixes=("_price", "_alt"),
            )
            delong_rows.append(
                {
                    "ticker": ticker,
                    "model_name": model_name,
                    "feature_set_alt": "price_plus_selected_alt",
                    **delong_auc_pvalue(
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
    feature_sets_path = EXPERIMENT_DIR / "feature_sets.json"
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
                "models": list(MODEL_DEFS.keys()),
                "feature_sets_by_ticker": feature_sets_by_ticker,
                "selection_manifest_path": str(selection_path or (SCREENING_DIR / "selected_feature_manifest.json")),
                "selection_note": (
                    "Selected alt features were screened using train-only statistics and walk-forward "
                    "validation before final holdout evaluation."
                ),
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
    result = run_selected_alt_final_models_experiment()
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
