from __future__ import annotations

import json
from pathlib import Path

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
EXPERIMENT_DIR = OUTPUT_DIR / "lag_feature_ablation"

LAG_STEPS = [1, 2, 3, 5]

PRICE_LAG_SOURCE_COLS = [
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
]

ALT_LAG_SOURCE_COLS = [
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


def add_group_lags(
    frame: pd.DataFrame,
    source_cols: list[str],
    lag_steps: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    lagged = frame.copy()
    available_cols = [col for col in source_cols if col in lagged.columns]
    created_cols: list[str] = []

    for col in available_cols:
        grouped = lagged.groupby("ticker", sort=False)[col]
        for lag_step in lag_steps:
            lag_col = f"{col}_lag{lag_step}"
            lagged[lag_col] = grouped.shift(lag_step)
            created_cols.append(lag_col)

    return lagged, created_cols


def build_feature_variants(
    panel: pd.DataFrame,
    lag_steps: list[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]], dict[str, list[str]]]:
    lag_steps = lag_steps or list(LAG_STEPS)

    feature_panel = build_feature_panel(panel)
    base_feature_cols = get_feature_sets(feature_panel)["price_plus_alt"]

    lagged_panel, price_lag_cols = add_group_lags(feature_panel, PRICE_LAG_SOURCE_COLS, lag_steps)
    lagged_panel, alt_lag_cols = add_group_lags(lagged_panel, ALT_LAG_SOURCE_COLS, lag_steps)

    feature_variants = {
        "base": list(base_feature_cols),
        "base_plus_price_lags": list(dict.fromkeys(base_feature_cols + price_lag_cols)),
        "base_plus_alt_lags": list(dict.fromkeys(base_feature_cols + alt_lag_cols)),
        "base_plus_all_lags": list(dict.fromkeys(base_feature_cols + price_lag_cols + alt_lag_cols)),
    }

    feature_variants_by_mode: dict[str, dict[str, list[str]]] = {}
    for variant_name, cols in feature_variants.items():
        feature_variants_by_mode[variant_name] = {
            "pooled": list(cols),
            "separate": [col for col in cols if col != "ticker_code"],
        }

    lag_metadata = {
        "lag_steps": list(lag_steps),
        "price_lag_source_cols": [col for col in PRICE_LAG_SOURCE_COLS if col in feature_panel.columns],
        "alt_lag_source_cols": [col for col in ALT_LAG_SOURCE_COLS if col in feature_panel.columns],
        "price_lag_feature_cols": price_lag_cols,
        "alt_lag_feature_cols": alt_lag_cols,
    }

    return lagged_panel, feature_variants_by_mode, lag_metadata


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


def run_lag_feature_ablation_experiment(dataset_path: Path | None = None) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in panel.columns:
        if col not in {"date", "ticker", "company_name", "y"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    feature_panel, feature_variants, lag_metadata = build_feature_variants(panel, lag_steps=LAG_STEPS)
    train_df, test_df, test_start = split_train_test(feature_panel)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_variant, feature_modes in feature_variants.items():
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
        ["training_mode", "split", "ticker", "model_name", "feature_variant"]
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["training_mode", "model_name", "feature_variant", "split", "ticker", "date"]
    ).reset_index(drop=True)

    metrics_path = EXPERIMENT_DIR / "metrics.csv"
    predictions_path = EXPERIMENT_DIR / "predictions.csv"
    variants_path = EXPERIMENT_DIR / "feature_variants_by_mode.json"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    variants_path.write_text(
        json.dumps(feature_variants, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "test_start": test_start.date().isoformat(),
                "models": list(MODEL_DEFS.keys()),
                "feature_variants": feature_variants,
                "training_modes": ["pooled", "separate"],
                "lag_metadata": lag_metadata,
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
        "feature_variants": feature_variants,
        "lag_metadata": lag_metadata,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "variants_path": variants_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    result = run_lag_feature_ablation_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Predictions: {result['predictions_path']}")
    print(f"Feature variants: {result['variants_path']}")
    print(
        result["metrics"]
        .query("split == 'test' and ticker == 'ALL'")
        .sort_values(["training_mode", "model_name", "feature_variant"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
