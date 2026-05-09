from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DATASET_FILENAME = "stock_panel_nine_tickers_session_aligned_full_history_adjusted_google_score_raw.csv"
BENCHMARK_FILENAME = "market-benchmark-data_qqq.csv"
DEFAULT_OUTPUT_PREFIX = (
    "stock_panel_nine_tickers_session_aligned_full_history_qqq_market_context_adjusted_google_score"
)

IDENTITY_COLUMNS = ["date", "ticker", "company_name", "close_stock_price", "stock_volume"]

QQQ_BENCHMARK_COLUMNS = [
    "qqq_close_price",
    "qqq_volume",
    "qqq_return_1d",
    "qqq_return_5d",
    "qqq_return_20d",
    "qqq_rolling_volatility_20d",
    "qqq_volume_zscore_20d",
]

QQQ_RELATIVE_COLUMNS = [
    "relative_to_qqq_return_1d",
    "relative_to_qqq_return_5d",
    "relative_to_qqq_return_20d",
    "relative_volatility_to_qqq_20d",
    "rolling_beta_to_qqq_60d",
    "qqq_residual_return_1d",
]

QQQ_MARKET_CONTEXT_COLUMNS = [*QQQ_BENCHMARK_COLUMNS, *QQQ_RELATIVE_COLUMNS]


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods if min_periods is not None else max(5, window // 2)
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


def default_base_dataset_path(project_root: Path) -> Path:
    return project_root / "data" / "datasets" / BASE_DATASET_FILENAME


def default_benchmark_path(project_root: Path) -> Path:
    return project_root / "data" / "equity_data" / BENCHMARK_FILENAME


def output_paths(project_root: Path, output_prefix: str) -> dict[str, Path]:
    output_dir = project_root / "data" / "datasets"
    return {
        "dataset": output_dir / f"{output_prefix}_raw.csv",
        "summary": output_dir / f"{output_prefix}_summary.csv",
        "source_audit": output_dir / f"{output_prefix}_source_audit.csv",
        "metadata": output_dir / f"{output_prefix}_metadata.json",
    }


def load_base_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base panel not found: {path}")

    frame = pd.read_csv(path, parse_dates=["date"])
    if "close_stock_price" not in frame.columns:
        if "stock_price" not in frame.columns:
            raise KeyError("Base panel must contain either 'stock_price' or 'close_stock_price'.")
        frame = frame.rename(columns={"stock_price": "close_stock_price"})

    missing_columns = [column for column in IDENTITY_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Base panel is missing required columns: {missing_columns}")

    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    numeric_columns = ["close_stock_price", "stock_volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_qqq_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"QQQ benchmark file not found: {path}")

    frame = pd.read_csv(path)
    if "date" not in frame.columns and "Date" in frame.columns:
        frame = frame.rename(columns={"Date": "date"})
    if "date" not in frame.columns:
        raise KeyError("QQQ benchmark file must contain a 'Date' or 'date' column.")

    rename_map = {
        "benchmark_close_price": "qqq_close_price",
        "benchmark_volume": "qqq_volume",
        "benchmark_return_1d": "qqq_return_1d",
        "benchmark_return_5d": "qqq_return_5d",
        "benchmark_return_20d": "qqq_return_20d",
        "benchmark_rolling_volatility_20d": "qqq_rolling_volatility_20d",
    }
    frame = frame.rename(columns=rename_map)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    required_columns = ["date", "qqq_close_price", "qqq_volume"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"QQQ benchmark file is missing required columns: {missing_columns}")

    numeric_columns = [column for column in ["qqq_close_price", "qqq_volume"] if column in frame.columns]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

    frame = frame.sort_values("date").reset_index(drop=True)
    frame["qqq_return_1d"] = pd.to_numeric(
        frame.get("qqq_return_1d", frame["qqq_close_price"].pct_change()),
        errors="coerce",
    )
    frame["qqq_return_5d"] = pd.to_numeric(
        frame.get("qqq_return_5d", frame["qqq_close_price"].pct_change(5)),
        errors="coerce",
    )
    frame["qqq_return_20d"] = pd.to_numeric(
        frame.get("qqq_return_20d", frame["qqq_close_price"].pct_change(20)),
        errors="coerce",
    )
    frame["qqq_rolling_volatility_20d"] = pd.to_numeric(
        frame.get("qqq_rolling_volatility_20d", frame["qqq_return_1d"].rolling(20, min_periods=10).std()),
        errors="coerce",
    )
    frame["qqq_volume_zscore_20d"] = rolling_zscore(frame["qqq_volume"], window=20)

    return frame[["date", *QQQ_BENCHMARK_COLUMNS]].sort_values("date").reset_index(drop=True)


def add_market_context_features(base_panel: pd.DataFrame, qqq_benchmark: pd.DataFrame) -> pd.DataFrame:
    merged = base_panel.merge(qqq_benchmark, on="date", how="left", validate="many_to_one")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    def add_ticker_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").copy()
        stock_return_1d = group["close_stock_price"].pct_change()
        stock_return_5d = group["close_stock_price"].pct_change(5)
        stock_return_20d = group["close_stock_price"].pct_change(20)
        stock_volatility_20d = stock_return_1d.rolling(20, min_periods=10).std()

        group["relative_to_qqq_return_1d"] = stock_return_1d - group["qqq_return_1d"]
        group["relative_to_qqq_return_5d"] = stock_return_5d - group["qqq_return_5d"]
        group["relative_to_qqq_return_20d"] = stock_return_20d - group["qqq_return_20d"]
        group["relative_volatility_to_qqq_20d"] = stock_volatility_20d - group["qqq_rolling_volatility_20d"]

        shifted_stock_return = stock_return_1d.shift(1)
        shifted_market_return = group["qqq_return_1d"].shift(1)
        rolling_market_variance = shifted_market_return.rolling(60, min_periods=30).var()
        rolling_covariance = shifted_stock_return.rolling(60, min_periods=30).cov(shifted_market_return)
        group["rolling_beta_to_qqq_60d"] = rolling_covariance / rolling_market_variance.replace(0.0, np.nan)
        group["qqq_residual_return_1d"] = (
            stock_return_1d - group["rolling_beta_to_qqq_60d"] * group["qqq_return_1d"]
        )

        return group

    with_context = pd.concat(
        [add_ticker_features(group) for _, group in merged.groupby("ticker", sort=False)],
        ignore_index=True,
    )
    with_context[QQQ_MARKET_CONTEXT_COLUMNS] = with_context[QQQ_MARKET_CONTEXT_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    passthrough_columns = [
        column
        for column in base_panel.columns
        if column not in {*IDENTITY_COLUMNS, "stock_price"}
    ]
    ordered_columns = [*IDENTITY_COLUMNS, *QQQ_MARKET_CONTEXT_COLUMNS, *passthrough_columns]
    return with_context[ordered_columns].sort_values(["date", "ticker"]).reset_index(drop=True)


def build_source_audit(
    *,
    base_dataset_path: Path,
    benchmark_path: Path,
    base_panel: pd.DataFrame,
    qqq_benchmark: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_name": "base_panel",
                "path": str(base_dataset_path),
                "rows": int(len(base_panel)),
                "date_min": base_panel["date"].min().date().isoformat(),
                "date_max": base_panel["date"].max().date().isoformat(),
                "n_dates": int(base_panel["date"].nunique()),
                "n_tickers": int(base_panel["ticker"].nunique()),
            },
            {
                "source_name": "qqq_benchmark",
                "path": str(benchmark_path),
                "rows": int(len(qqq_benchmark)),
                "date_min": qqq_benchmark["date"].min().date().isoformat(),
                "date_max": qqq_benchmark["date"].max().date().isoformat(),
                "n_dates": int(qqq_benchmark["date"].nunique()),
                "n_tickers": 1,
            },
            {
                "source_name": "output_panel",
                "path": "",
                "rows": int(len(panel)),
                "date_min": panel["date"].min().date().isoformat(),
                "date_max": panel["date"].max().date().isoformat(),
                "n_dates": int(panel["date"].nunique()),
                "n_tickers": int(panel["ticker"].nunique()),
            },
        ]
    )


def build_panel_summary(panel: pd.DataFrame) -> pd.DataFrame:
    per_ticker_rows = panel.groupby("ticker").size()
    records: list[dict[str, object]] = [
        {"metric": "rows", "value": int(len(panel))},
        {"metric": "tickers", "value": int(panel["ticker"].nunique())},
        {"metric": "date_min", "value": panel["date"].min().date().isoformat()},
        {"metric": "date_max", "value": panel["date"].max().date().isoformat()},
        {"metric": "min_rows_per_ticker", "value": int(per_ticker_rows.min())},
        {"metric": "max_rows_per_ticker", "value": int(per_ticker_rows.max())},
        {"metric": "qqq_market_context_columns", "value": len(QQQ_MARKET_CONTEXT_COLUMNS)},
    ]
    for column in QQQ_MARKET_CONTEXT_COLUMNS:
        records.append({"metric": f"{column}_null_rate", "value": float(panel[column].isna().mean())})
    return pd.DataFrame(records)


def build_metadata(
    *,
    output_dataset_path: Path,
    output_summary_path: Path,
    output_source_audit_path: Path,
    base_dataset_path: Path,
    benchmark_path: Path,
    panel: pd.DataFrame,
) -> dict[str, object]:
    return {
        "dataset": output_dataset_path.name,
        "summary": output_summary_path.name,
        "source_audit": output_source_audit_path.name,
        "base_dataset": str(base_dataset_path),
        "benchmark_dataset": str(benchmark_path),
        "benchmark": "QQQ",
        "rows": int(len(panel)),
        "tickers": sorted(panel["ticker"].dropna().unique().tolist()),
        "date_min": panel["date"].min().date().isoformat(),
        "date_max": panel["date"].max().date().isoformat(),
        "price_policy": "Uses the original adjusted close stock_price as close_stock_price; no OHLC rich-price features are added.",
        "market_context_columns": QQQ_MARKET_CONTEXT_COLUMNS,
        "relative_feature_policy": (
            "Relative return features use same-session stock and QQQ returns. "
            "rolling_beta_to_qqq_60d uses a shifted trailing 60-session window with at least 30 observations."
        ),
    }


def build_qqq_market_context_dataset(
    *,
    project_root: Path,
    base_dataset_path: Path | None = None,
    benchmark_path: Path | None = None,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> dict[str, object]:
    project_root = project_root.resolve()
    base_dataset_path = base_dataset_path or default_base_dataset_path(project_root)
    benchmark_path = benchmark_path or default_benchmark_path(project_root)
    paths = output_paths(project_root, output_prefix)

    base_panel = load_base_panel(base_dataset_path)
    qqq_benchmark = load_qqq_benchmark(benchmark_path)
    panel = add_market_context_features(base_panel, qqq_benchmark)
    summary = build_panel_summary(panel)
    source_audit = build_source_audit(
        base_dataset_path=base_dataset_path,
        benchmark_path=benchmark_path,
        base_panel=base_panel,
        qqq_benchmark=qqq_benchmark,
        panel=panel,
    )
    source_audit.loc[source_audit["source_name"].eq("output_panel"), "path"] = str(paths["dataset"])
    metadata = build_metadata(
        output_dataset_path=paths["dataset"],
        output_summary_path=paths["summary"],
        output_source_audit_path=paths["source_audit"],
        base_dataset_path=base_dataset_path,
        benchmark_path=benchmark_path,
        panel=panel,
    )

    paths["dataset"].parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(paths["dataset"], index=False)
    summary.to_csv(paths["summary"], index=False)
    source_audit.to_csv(paths["source_audit"], index=False)
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "panel": panel,
        "summary": summary,
        "source_audit": source_audit,
        "metadata": metadata,
        "output_dataset_path": paths["dataset"],
        "output_summary_path": paths["summary"],
        "output_source_audit_path": paths["source_audit"],
        "output_metadata_path": paths["metadata"],
    }
