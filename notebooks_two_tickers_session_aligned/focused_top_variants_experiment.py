from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from price_alt_baseline_experiment import locate_panel_dataset, split_train_test
from source_feature_count_sweep_experiment import (
    MODEL_DEFS,
    TARGET_TICKERS,
    build_feature_panel,
    load_panel,
)
from strict_source_ablation_experiment import _LEGACY


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "focused_top_variants_experiment"

FOCUSED_MODEL_FEATURES = {
    "logreg": ["price_top14", "price14_plus_google_top2"],
    "random_forest": ["price_top14", "price14_plus_reddit_top4"],
}


def bootstrap_metric_summary_safe(prediction_frame: pd.DataFrame) -> dict[str, object]:
    if "decision_threshold" not in prediction_frame.columns:
        prediction_frame = prediction_frame.copy()
        prediction_frame["decision_threshold"] = 0.5
    return _LEGACY.bootstrap_metric_summary(prediction_frame)


def metrics_from_subset(
    subset: pd.DataFrame,
    ticker: str,
    model_name: str,
    feature_set: str,
    split_name: str,
    n_features: int,
    train_rows_per_feature: float | None,
) -> dict[str, object]:
    metrics = _LEGACY.classification_metrics(
        subset["y_true"],
        subset["y_pred"].to_numpy(),
        subset["p_up"].to_numpy(),
    )
    return {
        "ticker": ticker,
        "model_name": model_name,
        "feature_set": feature_set,
        "split": split_name,
        "n_rows": int(len(subset)),
        "n_features": int(n_features),
        "train_rows_per_feature": (
            np.nan if train_rows_per_feature is None else float(train_rows_per_feature)
        ),
        **metrics,
    }


def build_delta_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    baseline_df = test_df[test_df["feature_set"] == "price_top14"].copy()
    compare_df = test_df[test_df["feature_set"] != "price_top14"].copy()
    merged = compare_df.merge(
        baseline_df[["ticker", "model_name", "accuracy", "balanced_accuracy", "roc_auc"]],
        on=["ticker", "model_name"],
        suffixes=("_variant", "_price"),
    )
    merged["delta_accuracy"] = merged["accuracy_variant"] - merged["accuracy_price"]
    merged["delta_balanced_accuracy"] = (
        merged["balanced_accuracy_variant"] - merged["balanced_accuracy_price"]
    )
    merged["delta_roc_auc"] = merged["roc_auc_variant"] - merged["roc_auc_price"]
    return merged.sort_values(["ticker", "model_name", "feature_set"]).reset_index(drop=True)


def compute_delong_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test_predictions = predictions_df[predictions_df["split"] == "test"].copy()

    for ticker in sorted(test_predictions["ticker"].unique()):
        ticker_test = test_predictions[test_predictions["ticker"] == ticker].copy()
        for model_name, feature_sets in FOCUSED_MODEL_FEATURES.items():
            if "price_top14" not in feature_sets or len(feature_sets) < 2:
                continue
            price_pred = ticker_test[
                (ticker_test["model_name"] == model_name)
                & (ticker_test["feature_set"] == "price_top14")
            ].copy()
            alt_feature_set = [name for name in feature_sets if name != "price_top14"][0]
            alt_pred = ticker_test[
                (ticker_test["model_name"] == model_name)
                & (ticker_test["feature_set"] == alt_feature_set)
            ].copy()
            if price_pred.empty or alt_pred.empty:
                continue
            merged = price_pred.merge(
                alt_pred,
                on=["date", "ticker", "split", "y_true"],
                suffixes=("_price", "_alt"),
            )
            rows.append(
                {
                    "ticker": ticker,
                    "model_name": model_name,
                    "feature_set_alt": alt_feature_set,
                    **_LEGACY.delong_auc_pvalue(
                        merged["y_true"].to_numpy(),
                        merged["p_up_price"].to_numpy(),
                        merged["p_up_alt"].to_numpy(),
                    ),
                }
            )

    for model_name, feature_sets in FOCUSED_MODEL_FEATURES.items():
        if "price_top14" not in feature_sets or len(feature_sets) < 2:
            continue
        price_pred = test_predictions[
            (test_predictions["model_name"] == model_name)
            & (test_predictions["feature_set"] == "price_top14")
        ].copy()
        alt_feature_set = [name for name in feature_sets if name != "price_top14"][0]
        alt_pred = test_predictions[
            (test_predictions["model_name"] == model_name)
            & (test_predictions["feature_set"] == alt_feature_set)
        ].copy()
        if price_pred.empty or alt_pred.empty:
            continue
        merged = price_pred.merge(
            alt_pred,
            on=["date", "ticker", "split", "y_true"],
            suffixes=("_price", "_alt"),
        )
        rows.append(
            {
                "ticker": "ALL",
                "model_name": model_name,
                "feature_set_alt": alt_feature_set,
                **_LEGACY.delong_auc_pvalue(
                    merged["y_true"].to_numpy(),
                    merged["p_up_price"].to_numpy(),
                    merged["p_up_alt"].to_numpy(),
                ),
            }
        )

    delong_df = pd.DataFrame(rows)
    if not delong_df.empty:
        delong_df = delong_df.sort_values(["ticker", "model_name"]).reset_index(drop=True)
    return delong_df


def run_focused_top_variants_experiment(
    dataset_path: Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    tickers = tickers or list(TARGET_TICKERS)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(dataset_path)
    panel = panel[panel["ticker"].isin(tickers)].copy()
    feature_panel, feature_sets, family_orders = build_feature_panel(panel)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, object]] = []

    for ticker in tickers:
        ticker_frame = feature_panel[feature_panel["ticker"] == ticker].copy()
        train_df, test_df, _ = split_train_test(ticker_frame)

        for model_name, selected_feature_sets in FOCUSED_MODEL_FEATURES.items():
            model_template = MODEL_DEFS[model_name]

            for feature_set_name in selected_feature_sets:
                feature_cols = feature_sets[feature_set_name]
                train_work = train_df.dropna(subset=["y_dir"]).copy()
                test_work = test_df.dropna(subset=["y_dir"]).copy()
                train_rows_per_feature = len(train_work) / max(len(feature_cols), 1)

                model = clone(model_template)
                model.fit(train_work[feature_cols], train_work["y_dir"].astype(int))

                for split_name, split_frame in [("train", train_work), ("test", test_work)]:
                    X_split = split_frame[feature_cols]
                    y_split = split_frame["y_dir"].astype(int)
                    proba = model.predict_proba(X_split)[:, 1]
                    pred = (proba >= 0.5).astype(int)

                    split_predictions = pd.DataFrame(
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
                            "decision_threshold": 0.5,
                        }
                    )
                    prediction_frames.append(split_predictions)

                    metrics_rows.append(
                        metrics_from_subset(
                            subset=split_predictions,
                            ticker=ticker,
                            model_name=model_name,
                            feature_set=feature_set_name,
                            split_name=split_name,
                            n_features=len(feature_cols),
                            train_rows_per_feature=train_rows_per_feature,
                        )
                    )

                    if split_name == "test":
                        bootstrap_rows.append(
                            {
                                "ticker": ticker,
                                "model_name": model_name,
                                "feature_set": feature_set_name,
                                "split": "test",
                                **bootstrap_metric_summary_safe(split_predictions),
                            }
                        )

    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["ticker", "model_name", "feature_set", "split", "date"]
    ).reset_index(drop=True)

    all_rows: list[dict[str, object]] = []
    all_bootstrap_rows: list[dict[str, object]] = []
    for split_name in ["train", "test"]:
        split_pred = predictions_df[predictions_df["split"] == split_name].copy()
        for model_name, selected_feature_sets in FOCUSED_MODEL_FEATURES.items():
            for feature_set_name in selected_feature_sets:
                subset = split_pred[
                    (split_pred["model_name"] == model_name)
                    & (split_pred["feature_set"] == feature_set_name)
                ].copy()
                if subset.empty:
                    continue
                all_rows.append(
                    metrics_from_subset(
                        subset=subset,
                        ticker="ALL",
                        model_name=model_name,
                        feature_set=feature_set_name,
                        split_name=split_name,
                        n_features=len(feature_sets[feature_set_name]),
                        train_rows_per_feature=None,
                    )
                )
                if split_name == "test":
                    all_bootstrap_rows.append(
                        {
                            "ticker": "ALL",
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "split": "test",
                            **bootstrap_metric_summary_safe(subset),
                        }
                    )

    metrics_df = pd.concat([pd.DataFrame(metrics_rows), pd.DataFrame(all_rows)], ignore_index=True)
    metrics_df = metrics_df.sort_values(
        ["split", "ticker", "model_name", "feature_set"]
    ).reset_index(drop=True)
    bootstrap_df = pd.DataFrame(bootstrap_rows + all_bootstrap_rows).sort_values(
        ["ticker", "model_name", "feature_set"]
    ).reset_index(drop=True)
    delta_df = build_delta_summary(metrics_df)
    delong_df = compute_delong_summary(predictions_df)

    metrics_path = EXPERIMENT_DIR / "metrics.csv"
    predictions_path = EXPERIMENT_DIR / "predictions.csv"
    bootstrap_path = EXPERIMENT_DIR / "bootstrap_summary.csv"
    delta_path = EXPERIMENT_DIR / "test_deltas_vs_price14.csv"
    delong_path = EXPERIMENT_DIR / "delong_vs_price14.csv"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    delong_df.to_csv(delong_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "tickers": tickers,
                "focused_model_features": FOCUSED_MODEL_FEATURES,
                "family_orders": family_orders,
                "comparison_goal": "focused comparison of best variants from source feature count sweep against price_top14",
                "bootstrap_samples": _LEGACY.BOOTSTRAP_SAMPLES,
                "bootstrap_block_length": _LEGACY.BOOTSTRAP_BLOCK_LENGTH,
                "note": "Models use the same simple fit/0.5 threshold setup as the source sweep; this notebook adds bootstrap and DeLong comparisons.",
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    return {
        "dataset_path": dataset_path,
        "feature_panel": feature_panel,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "bootstrap_summary": bootstrap_df,
        "test_deltas": delta_df,
        "delong_summary": delong_df,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "bootstrap_path": bootstrap_path,
        "delta_path": delta_path,
        "delong_path": delong_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    result = run_focused_top_variants_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Bootstrap: {result['bootstrap_path']}")
    print(f"DeLong: {result['delong_path']}")
    print(
        result["metrics"]
        .query("split == 'test'")
        .sort_values(["ticker", "model_name", "feature_set"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
