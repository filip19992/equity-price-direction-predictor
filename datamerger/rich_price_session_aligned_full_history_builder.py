from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datamerger import session_aligned_full_history_builder as base


ALIGNMENT_POLICY = base.ALIGNMENT_POLICY
DEFAULT_BACKFILL_SUFFIXES = base.DEFAULT_BACKFILL_SUFFIXES
DEFAULT_TICKERS = base.DEFAULT_TICKERS
NINE_TICKERS = base.NINE_TICKERS

RICH_STOCK_SOURCE_STEM = "stock-prices-rich-data"

RICH_STOCK_RAW_COLUMNS = [
    "open_stock_price",
    "high_stock_price",
    "low_stock_price",
    "close_stock_price",
    "adjusted_close_stock_price",
    "stock_volume",
]

RICH_STOCK_DERIVED_COLUMNS = [
    "close_return_1d",
    "intraday_return",
    "overnight_gap_return",
    "daily_high_low_range",
    "close_position_in_daily_range",
]

RICH_STOCK_COLUMNS = [*RICH_STOCK_RAW_COLUMNS, *RICH_STOCK_DERIVED_COLUMNS]

ORDERED_PANEL_COLUMNS = [
    "date",
    "ticker",
    "company_name",
    *RICH_STOCK_COLUMNS,
    *[
        column
        for column in base.ORDERED_PANEL_COLUMNS
        if column not in {"date", "ticker", "company_name", "stock_price", "stock_volume"}
    ],
]


def default_output_prefix(tickers: tuple[str, ...]) -> str:
    return f"{base.default_output_prefix(tickers)}_rich_price"


def build_rich_stock_candidates(
    ticker: str,
    backfill_suffixes: tuple[str, ...],
) -> tuple[str, ...]:
    ticker_tag = ticker.lower()
    candidates = [
        f"{RICH_STOCK_SOURCE_STEM}_{ticker_tag}_{suffix}.csv"
        for suffix in backfill_suffixes
    ]
    candidates.append(f"{RICH_STOCK_SOURCE_STEM}_{ticker_tag}.csv")
    return tuple(candidates)


def build_ticker_spec(
    ticker: str,
    backfill_suffixes: tuple[str, ...],
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "company_name": base.COMPANY_NAMES[ticker],
        "stock_candidates": build_rich_stock_candidates(ticker, backfill_suffixes),
        "google_candidates": base.build_source_candidates(ticker, "google_trends", backfill_suffixes),
        "reddit_subm_candidates": base.build_source_candidates(
            ticker,
            "reddit_submissions",
            backfill_suffixes,
        ),
        "reddit_comm_candidates": base.build_source_candidates(
            ticker,
            "reddit_comments",
            backfill_suffixes,
        ),
        "gdelt_candidates": base.build_source_candidates(ticker, "gdelt", backfill_suffixes),
    }


def normalize_rich_stock_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Open": "open_stock_price",
        "High": "high_stock_price",
        "Low": "low_stock_price",
        "Close": "close_stock_price",
        "Adj Close": "adjusted_close_stock_price",
        "Adj_Close": "adjusted_close_stock_price",
        "stock_price": "close_stock_price",
        "Volume": "stock_volume",
    }
    return frame.rename(columns=rename_map)


def add_price_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    frame = stock_df.sort_values("date").copy()
    previous_close = frame["close_stock_price"].shift(1)
    daily_range_denominator = frame["high_stock_price"] - frame["low_stock_price"]

    frame["close_return_1d"] = frame["close_stock_price"] / previous_close - 1.0
    frame["intraday_return"] = frame["close_stock_price"] / frame["open_stock_price"] - 1.0
    frame["overnight_gap_return"] = frame["open_stock_price"] / previous_close - 1.0
    frame["daily_high_low_range"] = frame["high_stock_price"] / frame["low_stock_price"] - 1.0
    frame["close_position_in_daily_range"] = (
        (frame["close_stock_price"] - frame["low_stock_price"]) / daily_range_denominator
    )
    frame.loc[daily_range_denominator.eq(0.0), "close_position_in_daily_range"] = np.nan
    return frame


def load_stock_frame(raw_dir: Path, spec: dict[str, object]) -> tuple[pd.DataFrame, list[Path]]:
    frame, paths = base.load_combined_csv_source(raw_dir, spec, "stock_candidates", "rich stock")
    frame = normalize_rich_stock_columns(frame)

    missing_required = [
        column
        for column in ["open_stock_price", "high_stock_price", "low_stock_price", "close_stock_price", "stock_volume"]
        if column not in frame.columns
    ]
    if missing_required:
        raise KeyError(
            f"Rich stock source for {spec['ticker']} is missing required columns: "
            + ", ".join(missing_required)
            + ". Run the stock_price_rich importer first."
        )
    if "adjusted_close_stock_price" not in frame.columns:
        frame["adjusted_close_stock_price"] = frame["close_stock_price"]

    frame = base.coerce_numeric_columns(frame, RICH_STOCK_RAW_COLUMNS)
    frame = add_price_features(frame[["date", *RICH_STOCK_RAW_COLUMNS]])
    frame = frame[["date", *RICH_STOCK_COLUMNS]].sort_values("date").reset_index(drop=True)
    return frame, paths


def inspect_source(
    raw_dir: Path,
    spec: dict[str, object],
    source_name: str,
    candidates_key: str,
) -> dict[str, object]:
    return base.inspect_source(raw_dir, spec, source_name, candidates_key)


def build_raw_source_audit(raw_dir: Path, ticker_specs: list[dict[str, object]]) -> pd.DataFrame:
    records = []
    source_map = {
        "rich_stock": "stock_candidates",
        "google_trends": "google_candidates",
        "reddit_submissions": "reddit_subm_candidates",
        "reddit_comments": "reddit_comm_candidates",
        "gdelt": "gdelt_candidates",
    }
    for spec in ticker_specs:
        for source_name, candidates_key in source_map.items():
            records.append(inspect_source(raw_dir, spec, source_name, candidates_key))
    return pd.DataFrame(records).sort_values(["ticker", "source_name"]).reset_index(drop=True)


def build_panel_for_ticker(raw_dir: Path, spec: dict[str, object]) -> pd.DataFrame:
    stock_df, _ = load_stock_frame(raw_dir, spec)
    google_df, _ = base.load_google_frame(raw_dir, spec)
    gdelt_df, _ = base.load_gdelt_frame(raw_dir, spec)
    subm_df, _ = base.load_reddit_daily_frame(raw_dir, spec, "reddit_subm_candidates")
    comm_df, _ = base.load_reddit_daily_frame(raw_dir, spec, "reddit_comm_candidates")

    panel = stock_df.copy()
    panel["ticker"] = spec["ticker"]
    panel["company_name"] = spec["company_name"]

    panel = panel.merge(base.prepare_reddit_for_merge(subm_df, "subm"), on="date", how="left")
    panel = panel.merge(base.prepare_reddit_for_merge(comm_df, "comm"), on="date", how="left")
    panel = panel.merge(base.align_google_to_sessions(google_df, stock_df["date"]), on="date", how="left")
    panel = panel.merge(base.align_gdelt_to_sessions(gdelt_df, stock_df["date"]), on="date", how="left")
    return panel[ORDERED_PANEL_COLUMNS].sort_values("date").reset_index(drop=True)


def build_panel(raw_dir: Path, ticker_specs: list[dict[str, object]]) -> pd.DataFrame:
    frames = [build_panel_for_ticker(raw_dir, spec) for spec in ticker_specs]
    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def build_panel_summary(panel: pd.DataFrame) -> pd.DataFrame:
    per_ticker_rows = panel.groupby("ticker").size()
    return pd.DataFrame(
        [
            {"metric": "rows", "value": int(len(panel))},
            {"metric": "tickers", "value": int(panel["ticker"].nunique())},
            {"metric": "date_min", "value": panel["date"].min().date().isoformat()},
            {"metric": "date_max", "value": panel["date"].max().date().isoformat()},
            {"metric": "min_rows_per_ticker", "value": int(per_ticker_rows.min())},
            {"metric": "max_rows_per_ticker", "value": int(per_ticker_rows.max())},
            {"metric": "rich_stock_columns", "value": len(RICH_STOCK_COLUMNS)},
            {"metric": "subm_finbert_null_rate", "value": float(panel["subm_reddit_finbert_mean"].isna().mean())},
            {"metric": "comm_finbert_null_rate", "value": float(panel["comm_reddit_finbert_mean"].isna().mean())},
        ]
    )


def build_metadata(
    *,
    output_dataset_path: Path,
    output_summary_path: Path,
    output_source_audit_path: Path,
    output_reddit_coverage_path: Path,
    source_audit: pd.DataFrame,
    reddit_coverage_audit: pd.DataFrame,
    panel: pd.DataFrame,
    backfill_suffixes: tuple[str, ...],
) -> dict[str, object]:
    return {
        "alignment_policy": ALIGNMENT_POLICY,
        "rows": int(len(panel)),
        "tickers_included": sorted(panel["ticker"].unique().tolist()),
        "date_min": panel["date"].min().date().isoformat(),
        "date_max": panel["date"].max().date().isoformat(),
        "backfill_suffixes": list(backfill_suffixes),
        "rich_stock_source_stem": RICH_STOCK_SOURCE_STEM,
        "rich_stock_columns": RICH_STOCK_COLUMNS,
        "output_path": str(output_dataset_path),
        "summary_path": str(output_summary_path),
        "source_audit_path": str(output_source_audit_path),
        "reddit_coverage_path": str(output_reddit_coverage_path),
        "notes": [
            "Raw source files are read-only inputs.",
            "This builder expects rich stock files from the stock_price_rich importer.",
            "The old stock_price column is intentionally replaced by close_stock_price.",
            "OHLC inputs are yfinance-adjusted prices from the stock_price_rich importer.",
            "close_stock_price is the replacement for the old stock_price column.",
            "OHLC inputs are preserved as open_stock_price, high_stock_price, low_stock_price, close_stock_price, and adjusted_close_stock_price.",
            "Simple same-day price features are added from OHLC values: close_return_1d, intraday_return, overnight_gap_return, daily_high_low_range, and close_position_in_daily_range.",
            "Alternative data source alignment matches the existing session_aligned_full_history_builder behavior.",
        ],
        "raw_source_rows": source_audit.to_dict(orient="records"),
        "coverage_rows": reddit_coverage_audit.to_dict(orient="records"),
    }


def build_rich_price_full_history_dataset(
    *,
    project_root: Path | None = None,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    backfill_suffixes: tuple[str, ...] = DEFAULT_BACKFILL_SUFFIXES,
    output_prefix: str | None = None,
) -> dict[str, object]:
    root = base.find_project_root(project_root)
    raw_dir = root / "data" / "equity_data"
    datasets_dir = root / "data" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    resolved_tickers = base.normalize_tickers(tickers)
    resolved_prefix = output_prefix or default_output_prefix(resolved_tickers)
    output_dataset_path = datasets_dir / f"{resolved_prefix}_raw.csv"
    output_summary_path = datasets_dir / f"{resolved_prefix}_summary.csv"
    output_metadata_path = datasets_dir / f"{resolved_prefix}_metadata.json"
    output_source_audit_path = datasets_dir / f"{resolved_prefix}_source_audit.csv"
    output_reddit_coverage_path = datasets_dir / f"{resolved_prefix}_reddit_coverage_audit.csv"

    ticker_specs = [
        build_ticker_spec(ticker, backfill_suffixes)
        for ticker in resolved_tickers
    ]
    source_audit = build_raw_source_audit(raw_dir, ticker_specs)
    reddit_coverage_audit = base.build_reddit_coverage_audit(raw_dir, ticker_specs)
    panel = build_panel(raw_dir, ticker_specs)
    summary = build_panel_summary(panel)
    metadata = build_metadata(
        output_dataset_path=output_dataset_path,
        output_summary_path=output_summary_path,
        output_source_audit_path=output_source_audit_path,
        output_reddit_coverage_path=output_reddit_coverage_path,
        source_audit=source_audit,
        reddit_coverage_audit=reddit_coverage_audit,
        panel=panel,
        backfill_suffixes=backfill_suffixes,
    )

    source_audit.to_csv(output_source_audit_path, index=False)
    reddit_coverage_audit.to_csv(output_reddit_coverage_path, index=False)
    panel.to_csv(output_dataset_path, index=False)
    summary.to_csv(output_summary_path, index=False)
    with output_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "panel": panel,
        "summary": summary,
        "source_audit": source_audit,
        "reddit_coverage_audit": reddit_coverage_audit,
        "metadata": metadata,
        "output_dataset_path": output_dataset_path,
        "output_summary_path": output_summary_path,
        "output_metadata_path": output_metadata_path,
        "output_source_audit_path": output_source_audit_path,
        "output_reddit_coverage_path": output_reddit_coverage_path,
    }


def parse_ticker_values(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_TICKERS
    tickers: list[str] = []
    for value in values:
        tickers.extend(part for part in value.replace(",", " ").split() if part)
    return base.normalize_tickers(tickers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Tickers to include. Defaults to all 10 imported big-tech tickers.",
    )
    parser.add_argument(
        "--nine-tickers",
        action="store_true",
        help="Use the original nine-ticker universe by excluding AVGO.",
    )
    parser.add_argument(
        "--backfill-suffix",
        action="append",
        dest="backfill_suffixes",
        help="Backfill filename suffix to merge, e.g. 20210101_20221231.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output prefix under data/datasets. Defaults from the ticker universe.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    tickers = NINE_TICKERS if args.nine_tickers else parse_ticker_values(args.tickers)
    backfill_suffixes = tuple(args.backfill_suffixes or DEFAULT_BACKFILL_SUFFIXES)
    result = build_rich_price_full_history_dataset(
        tickers=tickers,
        backfill_suffixes=backfill_suffixes,
        output_prefix=args.output_prefix,
    )
    print(f"Saved rich-price panel to {result['output_dataset_path']}")
    print(f"Saved summary to {result['output_summary_path']}")
    print(f"Saved metadata to {result['output_metadata_path']}")
    return result


if __name__ == "__main__":
    main()
