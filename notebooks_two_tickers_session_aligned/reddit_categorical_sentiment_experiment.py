from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from price_alt_baseline_experiment import (
    RANDOM_STATE,
    classification_metrics,
    engineer_ticker_features,
    get_feature_sets,
    locate_panel_dataset,
    rolling_zscore,
    split_train_test,
)
from strict_source_ablation_experiment import TARGET_TICKERS, load_panel


NOTEBOOK_DIR = Path(__file__).resolve().parent
CODE_DIR = NOTEBOOK_DIR.parent
DATA_DIR = CODE_DIR / "data" / "equity_data"
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "reddit_categorical_sentiment_experiment"

SENTIMENT_THRESHOLD = 0.05

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
}


def get_numeric_series(frame: pd.DataFrame, column: str, default: float | None = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def locate_reddit_raw_path(ticker: str, source_kind: str) -> Path:
    ticker_lower = ticker.lower()
    if source_kind == "subm":
        candidates = [
            DATA_DIR / f"stocks_posts_{ticker_lower}.parquet",
            DATA_DIR / f"{ticker_lower}_stocks_posts.parquet",
        ]
        if ticker.upper() == "TSLA":
            candidates.insert(0, DATA_DIR / "tesla_stocks_posts.parquet")
    elif source_kind == "comm":
        candidates = [
            DATA_DIR / f"stocks_comments_{ticker_lower}.parquet",
            DATA_DIR / f"{ticker_lower}_stocks_comments.parquet",
        ]
        if ticker.upper() == "TSLA":
            candidates.insert(0, DATA_DIR / "tesla_stocks_comments.parquet")
    else:
        raise ValueError("source_kind must be 'subm' or 'comm'")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Raw Reddit parquet not found for ticker={ticker}, source_kind={source_kind}")


def choose_sentiment_score(frame: pd.DataFrame) -> pd.Series:
    finbert = get_numeric_series(frame, "finbert_sentiment")
    vader = get_numeric_series(frame, "vader_sentiment")
    return finbert.where(finbert.notna(), vader)


def encode_sentiment_labels(score: pd.Series, threshold: float = SENTIMENT_THRESHOLD) -> pd.Series:
    labels = pd.Series(index=score.index, data=np.nan, dtype="float64")
    labels.loc[score.notna() & (score > threshold)] = 1.0
    labels.loc[score.notna() & (score < -threshold)] = -1.0
    labels.loc[score.notna() & score.between(-threshold, threshold, inclusive="both")] = 0.0
    return labels


def build_weighted_entropy(pos_share: pd.Series, neu_share: pd.Series, neg_share: pd.Series) -> pd.Series:
    probs = pd.concat([pos_share, neu_share, neg_share], axis=1).fillna(0.0)
    return -(probs * np.log(probs.where(probs > 0.0, 1.0))).sum(axis=1)


def aggregate_reddit_source_daily(ticker: str, source_kind: str) -> pd.DataFrame:
    path = locate_reddit_raw_path(ticker, source_kind)
    raw = pd.read_parquet(path)
    if raw.empty:
        return pd.DataFrame(columns=["date"])

    work = raw.copy()
    work["date"] = pd.to_datetime(work["aligned_date"], errors="coerce")
    work = work.dropna(subset=["date"]).reset_index(drop=True)

    work["chosen_sentiment"] = choose_sentiment_score(work)
    work["sentiment_label"] = encode_sentiment_labels(work["chosen_sentiment"])
    work["engagement_weight"] = get_numeric_series(work, "engagement_weight", default=1.0).fillna(1.0)
    work["engagement_weight"] = work["engagement_weight"].clip(lower=0.0)
    work["label_weight"] = work["engagement_weight"].where(work["sentiment_label"].notna(), 0.0)

    grouped = work.groupby("date", as_index=False)
    daily = grouped.agg(
        reddit_posts=("id", "count"),
        reddit_weight_sum=("engagement_weight", "sum"),
        reddit_labeled_weight_sum=("label_weight", "sum"),
    )

    pos_weight = work["label_weight"].where(work["sentiment_label"] == 1.0, 0.0).groupby(work["date"]).sum()
    neu_weight = work["label_weight"].where(work["sentiment_label"] == 0.0, 0.0).groupby(work["date"]).sum()
    neg_weight = work["label_weight"].where(work["sentiment_label"] == -1.0, 0.0).groupby(work["date"]).sum()
    weighted_sent_sum = (work["chosen_sentiment"] * work["label_weight"]).groupby(work["date"]).sum()

    daily = daily.set_index("date")
    daily["pos_weight"] = pos_weight
    daily["neu_weight"] = neu_weight
    daily["neg_weight"] = neg_weight
    daily["numeric_sent_sum"] = weighted_sent_sum

    denom = daily["reddit_labeled_weight_sum"].replace(0.0, np.nan)
    daily["numeric_sent_mean"] = daily["numeric_sent_sum"] / denom
    daily["pos_share"] = daily["pos_weight"] / denom
    daily["neu_share"] = daily["neu_weight"] / denom
    daily["neg_share"] = daily["neg_weight"] / denom
    daily["balance"] = daily["pos_share"] - daily["neg_share"]
    daily["entropy"] = build_weighted_entropy(daily["pos_share"], daily["neu_share"], daily["neg_share"])
    daily["majority_pos_flag"] = (
        (daily["pos_share"] > daily["neg_share"]) & (daily["pos_share"] > daily["neu_share"])
    ).astype(float)
    daily["majority_neg_flag"] = (
        (daily["neg_share"] > daily["pos_share"]) & (daily["neg_share"] > daily["neu_share"])
    ).astype(float)
    daily["active_flag"] = (daily["reddit_posts"] > 0).astype(float)
    daily["posts_log"] = np.log1p(daily["reddit_posts"].clip(lower=0.0))

    prefix = f"{source_kind}_"
    keep_cols = [
        "reddit_posts",
        "reddit_weight_sum",
        "numeric_sent_mean",
        "pos_share",
        "neu_share",
        "neg_share",
        "balance",
        "entropy",
        "majority_pos_flag",
        "majority_neg_flag",
        "active_flag",
        "posts_log",
    ]
    daily = daily[keep_cols].reset_index()
    daily = daily.rename(columns={col: f"{prefix}{col}" for col in keep_cols})
    return daily


def add_temporal_transforms(
    features: pd.DataFrame,
    source_name: str,
    series: pd.Series,
    *,
    include_current: bool = True,
    include_lag1: bool = True,
    include_roll3: bool = True,
    include_surprise20: bool = True,
) -> list[str]:
    created_cols: list[str] = []
    values = pd.to_numeric(series, errors="coerce")
    if include_current:
        col = f"{source_name}_current"
        features[col] = values
        created_cols.append(col)
    if include_lag1:
        col = f"{source_name}_lag1"
        features[col] = values.shift(1)
        created_cols.append(col)
    if include_roll3:
        col = f"{source_name}_roll3"
        features[col] = values.rolling(3, min_periods=2).mean()
        created_cols.append(col)
    if include_surprise20:
        col = f"{source_name}_surprise20"
        features[col] = rolling_zscore(values, 20)
        created_cols.append(col)
    return created_cols


def build_reddit_daily_feature_frame(panel: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = panel[panel["ticker"] == ticker][["date"]].copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")

    subm_daily = aggregate_reddit_source_daily(ticker, "subm")
    comm_daily = aggregate_reddit_source_daily(ticker, "comm")
    subm_daily["date"] = pd.to_datetime(subm_daily["date"], errors="coerce")
    comm_daily["date"] = pd.to_datetime(comm_daily["date"], errors="coerce")

    merged = base.merge(subm_daily, on="date", how="left").merge(comm_daily, on="date", how="left")
    for prefix in ["subm", "comm"]:
        posts_col = f"{prefix}_reddit_posts"
        weight_col = f"{prefix}_reddit_weight_sum"
        active_col = f"{prefix}_active_flag"
        majority_pos_col = f"{prefix}_majority_pos_flag"
        majority_neg_col = f"{prefix}_majority_neg_flag"
        posts_log_col = f"{prefix}_posts_log"

        if posts_col in merged:
            merged[posts_col] = merged[posts_col].fillna(0.0)
        if weight_col in merged:
            merged[weight_col] = merged[weight_col].fillna(0.0)
        if active_col in merged:
            merged[active_col] = merged[active_col].fillna(0.0)
        if majority_pos_col in merged:
            merged[majority_pos_col] = merged[majority_pos_col].fillna(0.0)
        if majority_neg_col in merged:
            merged[majority_neg_col] = merged[majority_neg_col].fillna(0.0)
        if posts_log_col in merged:
            merged[posts_log_col] = merged[posts_log_col].fillna(0.0)

    audit_rows = []
    for prefix in ["subm", "comm"]:
        audit_rows.append(
            {
                "ticker": ticker,
                "source": prefix,
                "pct_active_days": float((merged[f"{prefix}_active_flag"] > 0).mean()),
                "avg_posts": float(merged[f"{prefix}_reddit_posts"].mean()),
                "avg_weight_sum": float(merged[f"{prefix}_reddit_weight_sum"].mean()),
                "avg_pos_share": float(merged[f"{prefix}_pos_share"].mean()),
                "avg_neu_share": float(merged[f"{prefix}_neu_share"].mean()),
                "avg_neg_share": float(merged[f"{prefix}_neg_share"].mean()),
                "avg_numeric_sent_mean": float(merged[f"{prefix}_numeric_sent_mean"].mean()),
            }
        )
    return merged, pd.DataFrame(audit_rows)


def engineer_ticker_features_for_comparison(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    work = frame.sort_values("date").reset_index(drop=True).copy()
    base = engineer_ticker_features(work)
    price_cols = [col for col in get_feature_sets(base)["price_only"] if col != "ticker_code"]
    metadata_cols = ["date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir"]

    features = base[metadata_cols + price_cols].copy()
    feature_sets = {"price_only": list(price_cols)}

    reddit_daily, audit_df = build_reddit_daily_feature_frame(work, ticker=work["ticker"].iloc[0])
    work = work.merge(reddit_daily, on="date", how="left")

    activity_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for prefix in ["subm", "comm"]:
        activity_cols.extend(
            add_temporal_transforms(
                features,
                f"{prefix}_posts_log",
                work[f"{prefix}_posts_log"],
                include_current=True,
                include_lag1=True,
                include_roll3=True,
                include_surprise20=True,
            )
        )
        active_flag_col = f"{prefix}_active_flag_current"
        features[active_flag_col] = pd.to_numeric(work[f"{prefix}_active_flag"], errors="coerce").fillna(0.0)
        activity_cols.append(active_flag_col)

        numeric_cols.extend(
            add_temporal_transforms(
                features,
                f"{prefix}_numeric_sent",
                work[f"{prefix}_numeric_sent_mean"],
                include_current=True,
                include_lag1=True,
                include_roll3=True,
                include_surprise20=True,
            )
        )

        for base_name in ["pos_share", "neg_share", "balance", "entropy"]:
            categorical_cols.extend(
                add_temporal_transforms(
                    features,
                    f"{prefix}_{base_name}",
                    work[f"{prefix}_{base_name}"],
                    include_current=True,
                    include_lag1=True,
                    include_roll3=True,
                    include_surprise20=True,
                )
            )

        for flag_name in ["majority_pos_flag", "majority_neg_flag"]:
            col = f"{prefix}_{flag_name}_current"
            features[col] = pd.to_numeric(work[f"{prefix}_{flag_name}"], errors="coerce").fillna(0.0)
            categorical_cols.append(col)

    feature_sets["price_plus_reddit_numeric"] = list(dict.fromkeys(price_cols + activity_cols + numeric_cols))
    feature_sets["price_plus_reddit_categorical"] = list(
        dict.fromkeys(price_cols + activity_cols + categorical_cols)
    )
    return features, feature_sets, audit_df


def build_feature_panel(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    pieces = []
    feature_sets_agg: dict[str, list[str]] = {}
    audit_frames: list[pd.DataFrame] = []

    for _, ticker_frame in panel.groupby("ticker", sort=True):
        ticker_features, ticker_feature_sets, audit_df = engineer_ticker_features_for_comparison(ticker_frame)
        pieces.append(ticker_features)
        audit_frames.append(audit_df)
        for feature_set_name, cols in ticker_feature_sets.items():
            feature_sets_agg.setdefault(feature_set_name, [])
            feature_sets_agg[feature_set_name] = list(
                dict.fromkeys(feature_sets_agg[feature_set_name] + cols)
            )

    features = pd.concat(pieces, ignore_index=True)
    audit_df = pd.concat(audit_frames, ignore_index=True).sort_values(["ticker", "source"]).reset_index(drop=True)

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

    return features, feature_sets_agg, audit_df


def build_test_delta_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    price_df = test_df[test_df["feature_set"] == "price_only"].copy()
    compare_df = test_df[test_df["feature_set"] != "price_only"].copy()
    merged = compare_df.merge(
        price_df[["ticker", "model_name", "accuracy", "balanced_accuracy", "roc_auc"]],
        on=["ticker", "model_name"],
        suffixes=("_variant", "_price"),
    )
    merged["delta_accuracy"] = merged["accuracy_variant"] - merged["accuracy_price"]
    merged["delta_balanced_accuracy"] = (
        merged["balanced_accuracy_variant"] - merged["balanced_accuracy_price"]
    )
    merged["delta_roc_auc"] = merged["roc_auc_variant"] - merged["roc_auc_price"]
    return merged[
        [
            "ticker",
            "model_name",
            "feature_set",
            "accuracy_variant",
            "accuracy_price",
            "delta_accuracy",
            "balanced_accuracy_variant",
            "balanced_accuracy_price",
            "delta_balanced_accuracy",
            "roc_auc_variant",
            "roc_auc_price",
            "delta_roc_auc",
        ]
    ].sort_values(["ticker", "model_name", "feature_set"]).reset_index(drop=True)


def run_reddit_categorical_sentiment_experiment(
    dataset_path: Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    tickers = tickers or list(TARGET_TICKERS)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(dataset_path)
    panel = panel[panel["ticker"].isin(tickers)].copy()
    feature_panel, feature_sets, audit_df = build_feature_panel(panel)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        ticker_frame = feature_panel[feature_panel["ticker"] == ticker].copy()
        train_df, test_df, _ = split_train_test(ticker_frame)

        for feature_set_name, feature_cols in feature_sets.items():
            for model_name, model_template in MODEL_DEFS.items():
                model = clone(model_template)

                train_work = train_df.dropna(subset=["y_dir"]).copy()
                test_work = test_df.dropna(subset=["y_dir"]).copy()

                X_train = train_work[feature_cols]
                y_train = train_work["y_dir"].astype(int)
                X_test = test_work[feature_cols]
                y_test = test_work["y_dir"].astype(int)

                model.fit(X_train, y_train)

                for split_name, split_frame, X_split, y_split in [
                    ("train", train_work, X_train, y_train),
                    ("test", test_work, X_test, y_test),
                ]:
                    proba = model.predict_proba(X_split)[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    metrics = classification_metrics(y_split, pred, proba)
                    metric_rows.append(
                        {
                            "ticker": ticker,
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "split": split_name,
                            "n_rows": int(len(split_frame)),
                            "n_features": int(len(feature_cols)),
                            **metrics,
                        }
                    )
                    prediction_frames.append(
                        pd.DataFrame(
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
                            }
                        )
                    )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["split", "ticker", "model_name", "feature_set"]
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["split", "ticker", "model_name", "feature_set", "date"]
    ).reset_index(drop=True)

    all_rows: list[dict[str, object]] = []
    for split_name in ["train", "test"]:
        split_pred = predictions_df[predictions_df["split"] == split_name].copy()
        for model_name in MODEL_DEFS:
            for feature_set_name, feature_cols in feature_sets.items():
                subset = split_pred[
                    (split_pred["model_name"] == model_name)
                    & (split_pred["feature_set"] == feature_set_name)
                ].copy()
                if subset.empty:
                    continue
                metrics = classification_metrics(
                    subset["y_true"],
                    subset["y_pred"].to_numpy(),
                    subset["p_up"].to_numpy(),
                )
                all_rows.append(
                    {
                        "ticker": "ALL",
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "split": split_name,
                        "n_rows": int(len(subset)),
                        "n_features": int(len(feature_cols)),
                        **metrics,
                    }
                )

    metrics_df = pd.concat([metrics_df, pd.DataFrame(all_rows)], ignore_index=True).sort_values(
        ["split", "ticker", "model_name", "feature_set"]
    ).reset_index(drop=True)
    delta_df = build_test_delta_summary(metrics_df)

    metrics_path = EXPERIMENT_DIR / "metrics.csv"
    predictions_path = EXPERIMENT_DIR / "predictions.csv"
    audit_path = EXPERIMENT_DIR / "reddit_daily_audit.csv"
    delta_path = EXPERIMENT_DIR / "test_deltas_vs_price_only.csv"
    feature_sets_path = EXPERIMENT_DIR / "feature_sets.json"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    feature_sets_path.write_text(json.dumps(feature_sets, indent=2, ensure_ascii=True), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "tickers": tickers,
                "models": list(MODEL_DEFS.keys()),
                "feature_sets": feature_sets,
                "comparison_goal": "numeric Reddit sentiment versus categorical Reddit sentiment composition",
                "raw_reddit_level": "post/comment parquet aligned to trading session",
                "sentiment_threshold": SENTIMENT_THRESHOLD,
                "sentiment_score_rule": "use FinBERT when available, otherwise VADER",
                "categorical_features": [
                    "pos_share",
                    "neg_share",
                    "balance",
                    "entropy",
                    "majority_pos_flag",
                    "majority_neg_flag",
                ],
                "numeric_features": ["numeric_sent_mean"],
                "activity_features": ["posts_log", "active_flag"],
                "temporal_transforms": ["current", "lag1", "roll3", "surprise20"],
                "alt_time_alignment": "raw Reddit items are already aligned to next trading session in importer outputs",
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
        "reddit_daily_audit": audit_df,
        "test_deltas": delta_df,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "audit_path": audit_path,
        "delta_path": delta_path,
        "feature_sets_path": feature_sets_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    result = run_reddit_categorical_sentiment_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Audit: {result['audit_path']}")
    print(
        result["metrics"]
        .query("split == 'test'")
        .sort_values(["ticker", "model_name", "feature_set"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
