from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lag_feature_ablation_experiment import build_feature_variants
from price_alt_baseline_experiment import (
    MODEL_DEFS,
    classification_metrics,
    locate_panel_dataset,
    split_train_test,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "lag_model_comparison"

FEATURE_VARIANTS_TO_COMPARE = ["base", "base_plus_all_lags"]

MODEL_DEFS_WITH_MLP = {
    **MODEL_DEFS,
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
                    random_state=42,
                ),
            ),
        ]
    ),
}


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
    feature_variant: str,
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
                "feature_variant": feature_variant,
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
                    "feature_variant": feature_variant,
                    "split": split_name,
                    "ticker": ticker,
                    "n_rows": int(len(ticker_frame)),
                    "n_features": int(n_features),
                    **metrics_ticker,
                }
            )
    return rows


def run_lag_model_comparison_experiment(dataset_path: Path | None = None) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in panel.columns:
        if col not in {"date", "ticker", "company_name", "y"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    feature_panel, feature_variants, lag_metadata = build_feature_variants(panel)
    selected_variants = {
        name: feature_variants[name]
        for name in FEATURE_VARIANTS_TO_COMPARE
        if name in feature_variants
    }
    train_df, test_df, test_start = split_train_test(feature_panel)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_variant, feature_modes in selected_variants.items():
        for model_name, model_template in MODEL_DEFS_WITH_MLP.items():
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
            pooled_predictions["feature_variant"] = feature_variant
            prediction_frames.append(pooled_predictions)
            metrics_rows.extend(
                metrics_from_predictions(
                    prediction_frame=pooled_predictions,
                    training_mode="pooled",
                    model_name=model_name,
                    feature_variant=feature_variant,
                    n_features=len(pooled_feature_cols),
                )
            )

            separate_feature_cols = feature_modes["separate"]
            separate_predictions_parts = []
            for ticker in sorted(feature_panel["ticker"].dropna().unique()):
                train_ticker = train_df[train_df["ticker"] == ticker].copy()
                test_ticker = test_df[test_df["ticker"] == ticker].copy()
                train_pred, test_pred = fit_predict_frame(
                    model_template=model_template,
                    train_frame=train_ticker,
                    test_frame=test_ticker,
                    feature_cols=separate_feature_cols,
                )
                separate_predictions_parts.extend([train_pred, test_pred])

            separate_predictions = pd.concat(separate_predictions_parts, ignore_index=True)
            separate_predictions["training_mode"] = "separate"
            separate_predictions["model_name"] = model_name
            separate_predictions["feature_variant"] = feature_variant
            prediction_frames.append(separate_predictions)
            metrics_rows.extend(
                metrics_from_predictions(
                    prediction_frame=separate_predictions,
                    training_mode="separate",
                    model_name=model_name,
                    feature_variant=feature_variant,
                    n_features=len(separate_feature_cols),
                )
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["training_mode", "split", "ticker", "feature_variant", "model_name"]
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["training_mode", "feature_variant", "model_name", "split", "ticker", "date"]
    ).reset_index(drop=True)

    metrics_path = EXPERIMENT_DIR / "metrics.csv"
    predictions_path = EXPERIMENT_DIR / "predictions.csv"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "test_start": test_start.date().isoformat(),
                "feature_variants": selected_variants,
                "feature_variants_to_compare": FEATURE_VARIANTS_TO_COMPARE,
                "models": list(MODEL_DEFS_WITH_MLP.keys()),
                "training_modes": ["pooled", "separate"],
                "lag_metadata": lag_metadata,
                "mlp_config": {
                    "hidden_layer_sizes": [32, 16],
                    "alpha": 0.01,
                    "batch_size": 32,
                    "early_stopping": True,
                    "validation_fraction": 0.15,
                    "n_iter_no_change": 25,
                    "max_iter": 1000,
                    "random_state": 42,
                },
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
        "metadata_path": metadata_path,
    }


def main() -> None:
    result = run_lag_model_comparison_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Predictions: {result['predictions_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(
        result["metrics"]
        .query("split == 'test' and ticker == 'ALL'")
        .sort_values(["training_mode", "feature_variant", "model_name"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
