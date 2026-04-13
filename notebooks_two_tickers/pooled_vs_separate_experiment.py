from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from price_alt_baseline_experiment import (
    MODEL_DEFS,
    build_feature_panel,
    classification_metrics,
    get_feature_sets,
    locate_panel_dataset,
    split_train_test,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "pooled_vs_separate"


def normalize_feature_sets(feature_sets: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    normalized: dict[str, dict[str, list[str]]] = {}
    for feature_set_name, cols in feature_sets.items():
        pooled_cols = list(cols)
        separate_cols = [col for col in cols if col != "ticker_code"]
        normalized[feature_set_name] = {
            "pooled": pooled_cols,
            "separate": separate_cols,
        }
    return normalized


def fit_predict_frame(
    model_template,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_work = train_frame.dropna(subset=["y_dir"]).copy()
    test_work = test_frame.dropna(subset=["y_dir"]).copy()

    model = clone(model_template)
    model.fit(train_work[feature_cols], train_work["y_dir"].astype(int))

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
                    "y_true": split_frame["y_dir"].astype(int).values,
                    "y_pred": pred,
                    "p_up": proba,
                }
            )
        )

    return outputs[0], outputs[1]


def metrics_from_predictions(
    prediction_frame: pd.DataFrame,
    training_mode: str,
    model_name: str,
    feature_set_name: str,
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
                "training_mode": training_mode,
                "model_name": model_name,
                "feature_set": feature_set_name,
                "split": split_name,
                "ticker": "ALL",
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
                    "training_mode": training_mode,
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "split": split_name,
                    "ticker": ticker,
                    "n_rows": int(len(ticker_frame)),
                    "n_features": int(n_features),
                    **metrics_ticker,
                }
            )
    return rows


def run_pooled_vs_separate_experiment(dataset_path: Path | None = None) -> dict[str, object]:
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
    feature_sets = normalize_feature_sets(get_feature_sets(feature_panel))

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_set_name, feature_modes in feature_sets.items():
        for model_name, model_template in MODEL_DEFS.items():
            pooled_feature_cols = feature_modes["pooled"]
            pooled_train_pred, pooled_test_pred = fit_predict_frame(
                model_template=model_template,
                train_frame=train_df,
                test_frame=test_df,
                feature_cols=pooled_feature_cols,
            )
            pooled_predictions = pd.concat([pooled_train_pred, pooled_test_pred], ignore_index=True)
            pooled_predictions["training_mode"] = "pooled"
            pooled_predictions["model_name"] = model_name
            pooled_predictions["feature_set"] = feature_set_name
            prediction_frames.append(pooled_predictions)
            metrics_rows.extend(
                metrics_from_predictions(
                    prediction_frame=pooled_predictions,
                    training_mode="pooled",
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    n_features=len(pooled_feature_cols),
                )
            )

            separate_feature_cols = feature_modes["separate"]
            separate_predictions_per_ticker = []
            for ticker in sorted(feature_panel["ticker"].dropna().unique()):
                train_ticker = train_df[train_df["ticker"] == ticker].copy()
                test_ticker = test_df[test_df["ticker"] == ticker].copy()
                train_pred, test_pred = fit_predict_frame(
                    model_template=model_template,
                    train_frame=train_ticker,
                    test_frame=test_ticker,
                    feature_cols=separate_feature_cols,
                )
                separate_predictions_per_ticker.extend([train_pred, test_pred])

            separate_predictions = pd.concat(separate_predictions_per_ticker, ignore_index=True)
            separate_predictions["training_mode"] = "separate"
            separate_predictions["model_name"] = model_name
            separate_predictions["feature_set"] = feature_set_name
            prediction_frames.append(separate_predictions)
            metrics_rows.extend(
                metrics_from_predictions(
                    prediction_frame=separate_predictions,
                    training_mode="separate",
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    n_features=len(separate_feature_cols),
                )
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["training_mode", "split", "ticker", "model_name", "feature_set"]
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["training_mode", "model_name", "feature_set", "split", "ticker", "date"]
    ).reset_index(drop=True)

    metrics_path = EXPERIMENT_DIR / "metrics.csv"
    predictions_path = EXPERIMENT_DIR / "predictions.csv"
    feature_sets_path = EXPERIMENT_DIR / "feature_sets_by_mode.json"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"

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
                "models": list(MODEL_DEFS.keys()),
                "feature_sets": feature_sets,
                "training_modes": ["pooled", "separate"],
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
        "feature_sets_path": feature_sets_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    result = run_pooled_vs_separate_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Predictions: {result['predictions_path']}")
    print(f"Feature sets: {result['feature_sets_path']}")
    print(
        result["metrics"]
        .query("split == 'test' and ticker == 'ALL'")
        .sort_values(["training_mode", "model_name", "feature_set"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
