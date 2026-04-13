from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from price_alt_baseline_experiment import (
    RANDOM_STATE,
    build_feature_panel,
    classification_metrics,
    get_feature_sets,
    locate_panel_dataset,
    split_train_test,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "ticker_family_comparison"

TARGET_TICKERS = ["TSLA", "AAPL"]
FEATURE_SET_NAMES = ["price_only", "price_plus_alt"]


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


def fit_predict_frame(
    model_template,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_work = train_frame.dropna(subset=["y_dir"]).copy()
    test_work = test_frame.dropna(subset=["y_dir"]).copy()

    y_train = train_work["y_dir"].astype(int)
    if y_train.nunique() < 2:
        raise ValueError("Training target has fewer than 2 classes.")

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
                    "y_true": split_frame["y_dir"].astype(int).values,
                    "y_pred": pred,
                    "p_up": proba,
                }
            )
        )

    return outputs[0], outputs[1]


def metrics_from_predictions(
    prediction_frame: pd.DataFrame,
    ticker: str,
    model_name: str,
    feature_set_name: str,
    n_features: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
                **metrics,
            }
        )
    return rows


def load_feature_panel(dataset_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    for col in panel.columns:
        if col not in {"date", "ticker", "company_name", "y"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return build_feature_panel(panel)


def run_ticker_family_comparison_experiment(
    dataset_path: Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    tickers = tickers or list(TARGET_TICKERS)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    feature_panel = load_feature_panel(dataset_path)
    train_df, test_df, test_start = split_train_test(feature_panel)
    available_feature_sets = get_feature_sets(feature_panel)
    feature_sets = {
        name: available_feature_sets[name]
        for name in FEATURE_SET_NAMES
        if name in available_feature_sets
    }

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        train_ticker = train_df[train_df["ticker"] == ticker].copy()
        test_ticker = test_df[test_df["ticker"] == ticker].copy()
        if train_ticker.empty or test_ticker.empty:
            raise ValueError(f"Missing train or test rows for ticker {ticker}.")

        for feature_set_name, feature_cols in feature_sets.items():
            ticker_feature_cols = [col for col in feature_cols if col != "ticker_code"]
            for model_name, model_template in MODEL_DEFS.items():
                train_pred, test_pred = fit_predict_frame(
                    model_template=model_template,
                    train_frame=train_ticker,
                    test_frame=test_ticker,
                    feature_cols=ticker_feature_cols,
                )
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
                        n_features=len(ticker_feature_cols),
                    )
                )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["ticker", "split", "model_name", "feature_set"]
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["ticker", "model_name", "feature_set", "split", "date"]
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
                "feature_sets": feature_sets,
                "models": list(MODEL_DEFS.keys()),
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
        "feature_sets_path": feature_sets_path,
    }


def main() -> None:
    result = run_ticker_family_comparison_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Predictions: {result['predictions_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(
        result["metrics"]
        .query("split == 'test'")
        .sort_values(["ticker", "model_name", "feature_set"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
