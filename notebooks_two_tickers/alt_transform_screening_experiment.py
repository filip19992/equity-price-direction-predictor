from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from price_alt_baseline_experiment import (
    engineer_ticker_features,
    get_feature_sets,
    locate_panel_dataset,
    rolling_zscore,
    shifted_series,
    signed_log1p,
    split_train_test,
)
from strict_source_ablation_experiment import build_outer_folds, load_panel


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "alt_transform_screening"

TARGET_TICKERS = ["TSLA", "AAPL"]
SOURCE_ORDER = ["google_trends", "gdelt", "reddit_sentiment", "reddit_attention"]

MIN_NON_NULL_RATIO = 0.60
MIN_TRAIN_ROWS = 160
MIN_FOLD_COUNT = 3
MIN_TRAIN_AUC = 0.53
MIN_FOLD_MEAN_AUC = 0.53
MIN_FOLD_HIT_RATE = 0.75
TRAIN_AUC_PVALUE_THRESHOLD = 0.10
FALLBACK_MIN_FOLD_MEAN_AUC = 0.515
MAX_SELECTED_PER_SOURCE = 2
MAX_SELECTED_TOTAL = 8
CORR_PRUNE_THRESHOLD = 0.90


def average_available(series_list: list[pd.Series]) -> pd.Series:
    if not series_list:
        return pd.Series(dtype="float64")
    if len(series_list) == 1:
        return pd.to_numeric(series_list[0], errors="coerce")
    return pd.concat(series_list, axis=1).apply(pd.to_numeric, errors="coerce").mean(axis=1)


def create_transform_variants(series: pd.Series) -> dict[str, pd.Series]:
    values = pd.to_numeric(series, errors="coerce")
    roll3 = values.rolling(3, min_periods=2).mean()
    roll5 = values.rolling(5, min_periods=3).mean()
    surprise20 = rolling_zscore(values, 20)
    return {
        "level": values,
        "lag2": values.shift(1),
        "lag3": values.shift(2),
        "roll3": roll3,
        "roll5": roll5,
        "delta1": values.diff(1),
        "delta2": values.diff(2),
        "gap_roll3": values - roll3,
        "gap_roll5": values - roll5,
        "surprise20": surprise20,
        "spike20": (surprise20 > 1.0).astype(float),
    }


def build_raw_source_series(frame: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    trends_level = shifted_series(frame, "google_trends_score", fill="ffill")

    gdelt_articles = np.log1p(shifted_series(frame, "gdelt_articles", fill="ffill").clip(lower=0))
    gdelt_robust = shifted_series(frame, "gdelt_robust", fill="ffill")
    gdelt_sentiment = shifted_series(frame, "gdelt_sentiment_score", fill="ffill")

    subm_vader = shifted_series(frame, "subm_reddit_vader_weighted_mean", fill="zero")
    comm_vader = shifted_series(frame, "comm_reddit_vader_weighted_mean", fill="zero")
    subm_finbert = shifted_series(frame, "subm_reddit_finbert_weighted_mean", fill="zero")
    comm_finbert = shifted_series(frame, "comm_reddit_finbert_weighted_mean", fill="zero")
    reddit_vader = average_available([subm_vader, comm_vader])
    reddit_finbert = average_available([subm_finbert, comm_finbert])
    reddit_sentiment = average_available([reddit_vader, reddit_finbert])
    reddit_gap = (reddit_vader - reddit_finbert).abs()

    subm_posts = shifted_series(frame, "subm_reddit_posts", fill="zero")
    comm_posts = shifted_series(frame, "comm_reddit_posts", fill="zero")
    subm_comments = shifted_series(frame, "subm_reddit_comments_sum", fill="zero")
    comm_comments = shifted_series(frame, "comm_reddit_comments_sum", fill="zero")
    subm_score = shifted_series(frame, "subm_reddit_score_sum", fill="zero")
    comm_score = shifted_series(frame, "comm_reddit_score_sum", fill="zero")
    subm_weight = shifted_series(frame, "subm_reddit_weight_sum", fill="zero")
    comm_weight = shifted_series(frame, "comm_reddit_weight_sum", fill="zero")

    return {
        ("google_trends", "level"): trends_level,
        ("gdelt", "articles_log"): gdelt_articles,
        ("gdelt", "robust"): gdelt_robust,
        ("gdelt", "sentiment"): gdelt_sentiment,
        ("reddit_sentiment", "vader"): reddit_vader,
        ("reddit_sentiment", "finbert"): reddit_finbert,
        ("reddit_sentiment", "mean"): reddit_sentiment,
        ("reddit_sentiment", "gap"): reddit_gap,
        ("reddit_attention", "posts_total_log"): np.log1p((subm_posts + comm_posts).clip(lower=0)),
        ("reddit_attention", "comments_total_log"): np.log1p(
            (subm_comments + comm_comments).clip(lower=0)
        ),
        ("reddit_attention", "score_total_signed_log"): signed_log1p(subm_score + comm_score),
        ("reddit_attention", "weight_total_log"): np.log1p((subm_weight + comm_weight).clip(lower=0)),
    }


def build_candidate_feature_panel(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, str]] = []
    seen_features: set[str] = set()

    for _, ticker_frame in panel.groupby("ticker", sort=True):
        ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
        base = engineer_ticker_features(ticker_frame)
        price_cols = [
            col
            for col in get_feature_sets(base)["price_only"]
            if col != "ticker_code" and col in base.columns
        ]
        metadata_cols = ["date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir"]
        features = base[metadata_cols + price_cols].copy()
        generated_cols: dict[str, pd.Series] = {}

        for (source_group, raw_variable), raw_series in build_raw_source_series(ticker_frame).items():
            for transform_name, transformed in create_transform_variants(raw_series).items():
                feature_name = f"{source_group}__{raw_variable}__{transform_name}"
                generated_cols[feature_name] = transformed
                if feature_name not in seen_features:
                    metadata_rows.append(
                        {
                            "feature_name": feature_name,
                            "source_group": source_group,
                            "raw_variable": raw_variable,
                            "transform_name": transform_name,
                        }
                    )
                    seen_features.add(feature_name)

        if generated_cols:
            generated_df = pd.DataFrame(generated_cols, index=features.index)
            features = pd.concat([features, generated_df], axis=1, copy=False)

        pieces.append(features)

    feature_panel = pd.concat(pieces, ignore_index=True)
    feature_cols = [
        col
        for col in feature_panel.columns
        if col not in {"date", "ticker", "company_name", "future_return_1d", "y", "y_code", "y_dir"}
    ]
    constant_cols = [col for col in feature_cols if feature_panel[col].nunique(dropna=True) <= 1]
    if constant_cols:
        feature_panel = feature_panel.drop(columns=constant_cols)
        metadata_rows = [row for row in metadata_rows if row["feature_name"] not in constant_cols]

    candidate_catalog = pd.DataFrame(metadata_rows).sort_values(
        ["source_group", "raw_variable", "transform_name"]
    )
    return feature_panel, price_cols, candidate_catalog.reset_index(drop=True)


def auc_pvalue_vs_random(auc_value: float, positive_count: int, negative_count: int) -> float:
    if positive_count <= 0 or negative_count <= 0:
        return np.nan
    if not np.isfinite(auc_value):
        return np.nan

    auc_value = float(np.clip(auc_value, 1e-6, 1.0 - 1e-6))
    q1 = auc_value / (2.0 - auc_value)
    q2 = 2.0 * auc_value * auc_value / (1.0 + auc_value)
    variance = (
        auc_value * (1.0 - auc_value)
        + (positive_count - 1) * (q1 - auc_value * auc_value)
        + (negative_count - 1) * (q2 - auc_value * auc_value)
    ) / (positive_count * negative_count)
    if variance <= 0:
        return np.nan
    z_stat = (auc_value - 0.5) / math.sqrt(variance)
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0)))))


def impute_with_train_median(train_series: pd.Series, eval_series: pd.Series) -> tuple[np.ndarray, float]:
    median_value = float(pd.to_numeric(train_series, errors="coerce").median())
    if np.isnan(median_value):
        median_value = 0.0
    filled = pd.to_numeric(eval_series, errors="coerce").fillna(median_value).to_numpy(dtype="float64")
    return filled, median_value


def oriented_auc(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    direction: int | None = None,
) -> tuple[float, int]:
    y_arr = np.asarray(y_true, dtype="int64")
    score_arr = np.asarray(scores, dtype="float64")
    raw_auc = float(roc_auc_score(y_arr, score_arr))
    if direction is None:
        direction = 1 if raw_auc >= 0.5 else -1
    oriented = float(roc_auc_score(y_arr, score_arr * direction))
    return oriented, int(direction)


def evaluate_candidate_features(
    feature_panel: pd.DataFrame,
    candidate_catalog: pd.DataFrame,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for ticker in tickers:
        ticker_frame = feature_panel[feature_panel["ticker"] == ticker].copy()
        train_df, test_df, _ = split_train_test(ticker_frame)
        train_work = train_df.dropna(subset=["y_dir"]).copy()
        test_work = test_df.dropna(subset=["y_dir"]).copy()
        outer_folds = build_outer_folds(train_work)
        y_train = train_work["y_dir"].astype(int)
        y_test = test_work["y_dir"].astype(int)

        for candidate in candidate_catalog.itertuples(index=False):
            feature_name = candidate.feature_name
            if feature_name not in train_work.columns:
                continue

            train_series = train_work[feature_name]
            original_non_null_ratio = float(train_series.notna().mean())
            non_null_count = int(train_series.notna().sum())
            unique_non_null = int(train_series.nunique(dropna=True))

            if len(train_work) < MIN_TRAIN_ROWS or non_null_count < MIN_TRAIN_ROWS or unique_non_null < 2:
                metric_rows.append(
                    {
                        "ticker": ticker,
                        "feature_name": feature_name,
                        "source_group": candidate.source_group,
                        "raw_variable": candidate.raw_variable,
                        "transform_name": candidate.transform_name,
                        "train_rows": int(len(train_work)),
                        "train_non_null_ratio": original_non_null_ratio,
                        "train_non_null_count": non_null_count,
                        "train_auc": np.nan,
                        "train_auc_pvalue": np.nan,
                        "direction_sign": np.nan,
                        "fold_count": 0,
                        "fold_mean_auc": np.nan,
                        "fold_std_auc": np.nan,
                        "fold_min_auc": np.nan,
                        "fold_hit_rate": np.nan,
                        "test_auc": np.nan,
                        "selection_score": np.nan,
                    }
                )
                continue

            train_scores, _ = impute_with_train_median(train_series, train_series)
            train_auc, direction_sign = oriented_auc(y_train, train_scores)
            train_pvalue = auc_pvalue_vs_random(
                train_auc,
                int(y_train.sum()),
                int(len(y_train) - y_train.sum()),
            )

            test_scores, _ = impute_with_train_median(train_series, test_work[feature_name])
            test_auc, _ = oriented_auc(y_test, test_scores, direction=direction_sign)

            fold_aucs: list[float] = []
            for fold_idx, fold_train, fold_val in outer_folds:
                fold_train_work = fold_train.dropna(subset=["y_dir"]).copy()
                fold_val_work = fold_val.dropna(subset=["y_dir"]).copy()
                if fold_train_work.empty or fold_val_work.empty:
                    continue
                if fold_train_work["y_dir"].nunique() < 2 or fold_val_work["y_dir"].nunique() < 2:
                    continue
                if fold_train_work[feature_name].nunique(dropna=True) < 2:
                    continue

                fold_train_scores, _ = impute_with_train_median(
                    fold_train_work[feature_name],
                    fold_train_work[feature_name],
                )
                fold_val_scores, _ = impute_with_train_median(
                    fold_train_work[feature_name],
                    fold_val_work[feature_name],
                )
                fold_auc, fold_direction = oriented_auc(
                    fold_train_work["y_dir"].astype(int),
                    fold_train_scores,
                )
                val_auc, _ = oriented_auc(
                    fold_val_work["y_dir"].astype(int),
                    fold_val_scores,
                    direction=fold_direction,
                )
                fold_aucs.append(val_auc)
                fold_rows.append(
                    {
                        "ticker": ticker,
                        "fold": int(fold_idx),
                        "feature_name": feature_name,
                        "source_group": candidate.source_group,
                        "raw_variable": candidate.raw_variable,
                        "transform_name": candidate.transform_name,
                        "train_auc_fold": float(fold_auc),
                        "validation_auc_fold": float(val_auc),
                        "direction_sign_fold": int(fold_direction),
                        "n_train_rows": int(len(fold_train_work)),
                        "n_validation_rows": int(len(fold_val_work)),
                    }
                )

            fold_arr = np.asarray(fold_aucs, dtype="float64")
            fold_mean_auc = float(np.nanmean(fold_arr)) if fold_arr.size else np.nan
            fold_std_auc = float(np.nanstd(fold_arr, ddof=0)) if fold_arr.size else np.nan
            fold_min_auc = float(np.nanmin(fold_arr)) if fold_arr.size else np.nan
            fold_hit_rate = float(np.mean(fold_arr > 0.5)) if fold_arr.size else np.nan

            p_component = max(0.0, -math.log10(max(train_pvalue, 1e-12))) if np.isfinite(train_pvalue) else 0.0
            selection_score = (
                max(train_auc - 0.5, 0.0)
                * max(fold_mean_auc - 0.5, 0.0)
                * (fold_hit_rate if np.isfinite(fold_hit_rate) else 0.0)
                * max(p_component, 1.0)
            )

            metric_rows.append(
                {
                    "ticker": ticker,
                    "feature_name": feature_name,
                    "source_group": candidate.source_group,
                    "raw_variable": candidate.raw_variable,
                    "transform_name": candidate.transform_name,
                    "train_rows": int(len(train_work)),
                    "train_non_null_ratio": original_non_null_ratio,
                    "train_non_null_count": non_null_count,
                    "train_auc": float(train_auc),
                    "train_auc_pvalue": float(train_pvalue) if np.isfinite(train_pvalue) else np.nan,
                    "direction_sign": int(direction_sign),
                    "fold_count": int(len(fold_arr)),
                    "fold_mean_auc": fold_mean_auc,
                    "fold_std_auc": fold_std_auc,
                    "fold_min_auc": fold_min_auc,
                    "fold_hit_rate": fold_hit_rate,
                    "test_auc": float(test_auc),
                    "selection_score": float(selection_score),
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["ticker", "source_group", "raw_variable", "transform_name"]
    )
    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        fold_df = fold_df.sort_values(
            ["ticker", "fold", "source_group", "raw_variable", "transform_name"]
        )
    return metrics_df.reset_index(drop=True), fold_df.reset_index(drop=True)


def correlation_prune(
    ticker: str,
    candidate_rows: pd.DataFrame,
    feature_panel: pd.DataFrame,
) -> list[str]:
    if candidate_rows.empty:
        return []

    ticker_frame = feature_panel[feature_panel["ticker"] == ticker].copy()
    train_df, _, _ = split_train_test(ticker_frame)
    train_work = train_df.dropna(subset=["y_dir"]).copy()

    selected: list[str] = []
    for row in candidate_rows.itertuples(index=False):
        feature_name = row.feature_name
        series = pd.to_numeric(train_work[feature_name], errors="coerce")
        median_value = float(series.median()) if series.notna().any() else 0.0
        candidate_values = series.fillna(median_value)

        keep = True
        for selected_feature in selected:
            selected_series = pd.to_numeric(train_work[selected_feature], errors="coerce")
            selected_median = float(selected_series.median()) if selected_series.notna().any() else 0.0
            selected_values = selected_series.fillna(selected_median)
            corr = candidate_values.corr(selected_values)
            if np.isfinite(corr) and abs(float(corr)) >= CORR_PRUNE_THRESHOLD:
                keep = False
                break

        if keep:
            selected.append(feature_name)
        if len(selected) >= MAX_SELECTED_TOTAL:
            break

    return selected


def select_features(
    metrics_df: pd.DataFrame,
    feature_panel: pd.DataFrame,
    price_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = metrics_df.copy()
    rows["eligible_strict"] = (
        (rows["train_non_null_ratio"] >= MIN_NON_NULL_RATIO)
        & (rows["train_auc"] >= MIN_TRAIN_AUC)
        & (rows["train_auc_pvalue"] <= TRAIN_AUC_PVALUE_THRESHOLD)
        & (rows["fold_count"] >= MIN_FOLD_COUNT)
        & (rows["fold_mean_auc"] >= MIN_FOLD_MEAN_AUC)
        & (rows["fold_hit_rate"] >= MIN_FOLD_HIT_RATE)
    )

    rows["selected_candidate"] = False
    rows["selected_final"] = False
    rows["selection_reason"] = ""
    rows["selection_rank"] = np.nan

    manifest: dict[str, object] = {
        "selection_thresholds": {
            "min_non_null_ratio": MIN_NON_NULL_RATIO,
            "min_train_rows": MIN_TRAIN_ROWS,
            "min_fold_count": MIN_FOLD_COUNT,
            "min_train_auc": MIN_TRAIN_AUC,
            "min_fold_mean_auc": MIN_FOLD_MEAN_AUC,
            "min_fold_hit_rate": MIN_FOLD_HIT_RATE,
            "train_auc_pvalue_threshold": TRAIN_AUC_PVALUE_THRESHOLD,
            "fallback_min_fold_mean_auc": FALLBACK_MIN_FOLD_MEAN_AUC,
            "max_selected_per_source": MAX_SELECTED_PER_SOURCE,
            "max_selected_total": MAX_SELECTED_TOTAL,
            "corr_prune_threshold": CORR_PRUNE_THRESHOLD,
        },
        "tickers": {},
    }

    for ticker in sorted(rows["ticker"].unique()):
        ticker_rows = rows[rows["ticker"] == ticker].copy()
        source_candidates: list[pd.DataFrame] = []

        for source_group in SOURCE_ORDER:
            source_rows = ticker_rows[ticker_rows["source_group"] == source_group].copy()
            if source_rows.empty:
                continue

            source_rows = source_rows.sort_values(
                ["selection_score", "fold_mean_auc", "train_auc", "test_auc"],
                ascending=[False, False, False, False],
            )
            strict_rows = source_rows[source_rows["eligible_strict"]].drop_duplicates(
                subset=["raw_variable"],
                keep="first",
            )

            if not strict_rows.empty:
                chosen = strict_rows.head(MAX_SELECTED_PER_SOURCE).copy()
                chosen["selection_reason"] = "strict"
            else:
                fallback_rows = source_rows[source_rows["fold_mean_auc"] >= FALLBACK_MIN_FOLD_MEAN_AUC]
                fallback_rows = fallback_rows.drop_duplicates(subset=["raw_variable"], keep="first")
                chosen = fallback_rows.head(1).copy() if not fallback_rows.empty else pd.DataFrame()
                if not chosen.empty:
                    chosen["selection_reason"] = "fallback"

            if not chosen.empty:
                source_candidates.append(chosen)

        selected_pool = (
            pd.concat(source_candidates, ignore_index=True)
            .sort_values(
                ["selection_score", "fold_mean_auc", "train_auc", "test_auc"],
                ascending=[False, False, False, False],
            )
            .reset_index(drop=True)
            if source_candidates
            else pd.DataFrame(columns=rows.columns)
        )

        selected_final = correlation_prune(ticker, selected_pool, feature_panel)
        source_map: dict[str, list[str]] = {}
        selected_alt_rows = selected_pool[selected_pool["feature_name"].isin(selected_final)].copy()
        if not selected_alt_rows.empty:
            selected_alt_rows["selection_rank"] = range(1, len(selected_alt_rows) + 1)

        for source_group, group in selected_alt_rows.groupby("source_group", sort=False):
            source_map[source_group] = group["feature_name"].tolist()

        if not selected_pool.empty:
            candidate_index = rows.index[
                (rows["ticker"] == ticker)
                & (rows["feature_name"].isin(selected_pool["feature_name"]))
            ]
            rows.loc[candidate_index, "selected_candidate"] = True
            for _, candidate_row in selected_pool.iterrows():
                mask = (rows["ticker"] == ticker) & (rows["feature_name"] == candidate_row["feature_name"])
                rows.loc[mask, "selection_reason"] = candidate_row["selection_reason"]

        if selected_final:
            final_index = rows.index[
                (rows["ticker"] == ticker)
                & (rows["feature_name"].isin(selected_final))
            ]
            rows.loc[final_index, "selected_final"] = True
            rank_map = {feature_name: rank for rank, feature_name in enumerate(selected_final, start=1)}
            for feature_name, rank in rank_map.items():
                mask = (rows["ticker"] == ticker) & (rows["feature_name"] == feature_name)
                rows.loc[mask, "selection_rank"] = rank

        manifest["tickers"][ticker] = {
            "price_only_features": list(price_cols),
            "selected_alt_features": list(selected_final),
            "selected_by_source": source_map,
        }

    selected_df = rows.sort_values(
        ["ticker", "selected_final", "selected_candidate", "selection_rank", "selection_score"],
        ascending=[True, False, False, True, False],
    ).reset_index(drop=True)
    return selected_df, manifest


def run_alt_transform_screening_experiment(
    dataset_path: Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path or locate_panel_dataset()
    tickers = tickers or list(TARGET_TICKERS)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(dataset_path)
    panel = panel[panel["ticker"].isin(tickers)].copy()

    feature_panel, price_cols, candidate_catalog = build_candidate_feature_panel(panel)
    metrics_df, fold_df = evaluate_candidate_features(feature_panel, candidate_catalog, tickers)
    selected_df, selection_manifest = select_features(metrics_df, feature_panel, price_cols)

    metrics_path = EXPERIMENT_DIR / "feature_metrics.csv"
    fold_path = EXPERIMENT_DIR / "feature_fold_metrics.csv"
    catalog_path = EXPERIMENT_DIR / "candidate_feature_catalog.csv"
    selected_path = EXPERIMENT_DIR / "selected_features.csv"
    manifest_path = EXPERIMENT_DIR / "selected_feature_manifest.json"
    metadata_path = EXPERIMENT_DIR / "run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    candidate_catalog.to_csv(catalog_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    manifest_path.write_text(json.dumps(selection_manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "tickers": tickers,
                "price_only_feature_count": len(price_cols),
                "candidate_feature_count": int(len(candidate_catalog)),
                "selection_method": (
                    "Features are ranked on train-only statistics and walk-forward validation AUC, "
                    "then pruned for redundancy before final-model training."
                ),
                "time_alignment": "All alternative series are shifted by one trading day before transforms.",
                "transform_set": [
                    "level(t-1)",
                    "lag2(t-2)",
                    "lag3(t-3)",
                    "roll3",
                    "roll5",
                    "delta1",
                    "delta2",
                    "gap_roll3",
                    "gap_roll5",
                    "surprise20",
                    "spike20",
                ],
                "selection_thresholds": selection_manifest["selection_thresholds"],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    return {
        "dataset_path": dataset_path,
        "feature_panel": feature_panel,
        "candidate_catalog": candidate_catalog,
        "feature_metrics": metrics_df,
        "feature_fold_metrics": fold_df,
        "selected_features": selected_df,
        "selection_manifest": selection_manifest,
        "metrics_path": metrics_path,
        "fold_path": fold_path,
        "catalog_path": catalog_path,
        "selected_path": selected_path,
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    result = run_alt_transform_screening_experiment()
    print(f"Dataset: {result['dataset_path']}")
    print(f"Feature metrics: {result['metrics_path']}")
    print(f"Selected features: {result['selected_path']}")
    print(
        result["selected_features"]
        .query("selected_final")
        .sort_values(["ticker", "selection_rank"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
