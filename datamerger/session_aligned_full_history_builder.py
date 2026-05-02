from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ALIGNMENT_POLICY = "calendar-day alternative data mapped to same or next trading session"
DEFAULT_BACKFILL_SUFFIXES = ("20210101_20221231",)

COMPANY_NAMES = {
    "AAPL": "Apple",
    "AMD": "Advanced Micro Devices",
    "AMZN": "Amazon",
    "AVGO": "Broadcom",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "MSFT": "Microsoft",
    "NFLX": "Netflix",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
}

DEFAULT_TICKERS = (
    "AAPL",
    "AMD",
    "AMZN",
    "AVGO",
    "GOOGL",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "TSLA",
)
NINE_TICKERS = tuple(ticker for ticker in DEFAULT_TICKERS if ticker != "AVGO")

SOURCE_DEFINITIONS = {
    "stock": {
        "stem": "stock-prices-data",
        "legacy_name": "stock-prices-data.csv",
    },
    "google_trends": {
        "stem": "google_trends_data",
        "legacy_name": "google_trends_data.csv",
    },
    "reddit_submissions": {
        "stem": "stock-reddit-data",
        "legacy_name": "stock-reddit-data.csv",
    },
    "reddit_comments": {
        "stem": "stock-reddit-comments-data",
        "legacy_name": "stock-reddit-comments-data.csv",
    },
    "gdelt": {
        "stem": "gdelt_data",
        "legacy_name": "gdelt_data.csv",
    },
}

REDDIT_BASE_COLUMNS = [
    "reddit_posts",
    "reddit_weight_sum",
    "reddit_score_sum",
    "reddit_comments_sum",
    "reddit_vader_mean",
    "reddit_vader_sum",
    "reddit_vader_std",
    "reddit_finbert_mean",
    "reddit_finbert_sum",
    "reddit_finbert_std",
    "reddit_vader_weighted_mean",
    "reddit_finbert_weighted_mean",
    "reddit_sent_mean",
    "reddit_sent_sum",
    "reddit_sent_std",
]

GDELT_NUMERIC_COLUMNS = [
    "gdelt_articles",
    "gdelt_robust",
    "gdelt_sentiment_score",
]

ORDERED_PANEL_COLUMNS = [
    "date",
    "ticker",
    "company_name",
    "stock_price",
    "stock_volume",
    "subm_reddit_posts",
    "subm_reddit_weight_sum",
    "subm_reddit_score_sum",
    "subm_reddit_comments_sum",
    "subm_reddit_vader_mean",
    "subm_reddit_vader_sum",
    "subm_reddit_vader_std",
    "subm_reddit_finbert_mean",
    "subm_reddit_finbert_sum",
    "subm_reddit_finbert_std",
    "subm_reddit_vader_weighted_mean",
    "subm_reddit_finbert_weighted_mean",
    "subm_reddit_sent_mean",
    "subm_reddit_sent_sum",
    "subm_reddit_sent_std",
    "comm_reddit_posts",
    "comm_reddit_weight_sum",
    "comm_reddit_score_sum",
    "comm_reddit_comments_sum",
    "comm_reddit_vader_mean",
    "comm_reddit_vader_sum",
    "comm_reddit_vader_std",
    "comm_reddit_finbert_mean",
    "comm_reddit_finbert_sum",
    "comm_reddit_finbert_std",
    "comm_reddit_vader_weighted_mean",
    "comm_reddit_finbert_weighted_mean",
    "comm_reddit_sent_mean",
    "comm_reddit_sent_sum",
    "comm_reddit_sent_std",
    "google_trends_score",
    "gdelt_articles",
    "gdelt_robust",
    "gdelt_sentiment_score",
]


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "data" / "equity_data").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing data/equity_data.")


def normalize_tickers(tickers: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in tickers:
        ticker = raw.strip().upper()
        if not ticker or ticker in seen:
            continue
        if ticker not in COMPANY_NAMES:
            raise ValueError(f"Unsupported ticker: {ticker}")
        seen.add(ticker)
        normalized.append(ticker)
    return tuple(normalized)


def default_output_prefix(tickers: tuple[str, ...]) -> str:
    ticker_set = set(tickers)
    if ticker_set == set(DEFAULT_TICKERS):
        return "stock_panel_big_tech_10_session_aligned_full_history"
    if ticker_set == set(NINE_TICKERS):
        return "stock_panel_nine_tickers_session_aligned_full_history"
    return f"stock_panel_{len(tickers)}_tickers_session_aligned_full_history"


def build_source_candidates(
    ticker: str,
    source_name: str,
    backfill_suffixes: tuple[str, ...],
) -> tuple[str, ...]:
    definition = SOURCE_DEFINITIONS[source_name]
    stem = definition["stem"]
    ticker_tag = ticker.lower()

    candidates = [
        f"{stem}_{ticker_tag}_{suffix}.csv"
        for suffix in backfill_suffixes
    ]
    candidates.append(f"{stem}_{ticker_tag}.csv")

    # Older TSLA imports used generic names. Keep them as a fallback and let
    # duplicate dates from those files override the ticker-specific candidate.
    if ticker == "TSLA":
        candidates.append(definition["legacy_name"])

    return tuple(candidates)


def build_ticker_spec(
    ticker: str,
    backfill_suffixes: tuple[str, ...],
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "company_name": COMPANY_NAMES[ticker],
        "stock_candidates": build_source_candidates(ticker, "stock", backfill_suffixes),
        "google_candidates": build_source_candidates(ticker, "google_trends", backfill_suffixes),
        "reddit_subm_candidates": build_source_candidates(
            ticker, "reddit_submissions", backfill_suffixes
        ),
        "reddit_comm_candidates": build_source_candidates(
            ticker, "reddit_comments", backfill_suffixes
        ),
        "gdelt_candidates": build_source_candidates(ticker, "gdelt", backfill_suffixes),
    }


def existing_paths(raw_dir: Path, candidates: tuple[str, ...]) -> list[Path]:
    return [raw_dir / candidate for candidate in candidates if (raw_dir / candidate).exists()]


def read_csv_with_standard_date(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_column = next((col for col in ("date", "Date") if col in frame.columns), None)
    if date_column is None:
        raise KeyError(f"No supported date column found in {path}")
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"]).copy()
    return frame


def coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    working = frame.copy()
    for column in columns:
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")
    return working


def load_combined_csv_source(
    raw_dir: Path,
    spec: dict[str, object],
    source_key: str,
    label: str,
) -> tuple[pd.DataFrame, list[Path]]:
    candidates = tuple(spec[source_key])
    paths = existing_paths(raw_dir, candidates)
    if not paths:
        raise FileNotFoundError(
            f"No {label} source found for {spec['ticker']}. "
            f"Checked: {', '.join(candidates)}"
        )

    frames = []
    for order, path in enumerate(paths):
        frame = read_csv_with_standard_date(path)
        frame["_source_file"] = path.name
        frame["_source_order"] = order
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values(["date", "_source_order"])
        .drop_duplicates(subset=["date"], keep="last")
        .drop(columns=["_source_file", "_source_order"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return combined, paths


def load_stock_frame(raw_dir: Path, spec: dict[str, object]) -> tuple[pd.DataFrame, list[Path]]:
    frame, paths = load_combined_csv_source(raw_dir, spec, "stock_candidates", "stock")
    frame = coerce_numeric_columns(frame, ["stock_price", "stock_volume"])
    frame = frame[["date", "stock_price", "stock_volume"]].sort_values("date").reset_index(drop=True)
    return frame, paths


def load_google_frame(raw_dir: Path, spec: dict[str, object]) -> tuple[pd.DataFrame, list[Path]]:
    frame, paths = load_combined_csv_source(
        raw_dir, spec, "google_candidates", "Google Trends"
    )
    frame = frame.rename(columns={"trends_score": "google_trends_score"})
    frame = coerce_numeric_columns(frame, ["google_trends_score"])
    frame = frame[["date", "google_trends_score"]].sort_values("date").reset_index(drop=True)
    return frame, paths


def load_gdelt_frame(raw_dir: Path, spec: dict[str, object]) -> tuple[pd.DataFrame, list[Path]]:
    frame, paths = load_combined_csv_source(raw_dir, spec, "gdelt_candidates", "GDELT")
    frame = frame.rename(columns={"sentiment_score": "gdelt_sentiment_score"})
    frame = coerce_numeric_columns(frame, GDELT_NUMERIC_COLUMNS)
    frame = frame[["date", "gdelt_articles", "gdelt_robust", "gdelt_sentiment_score"]]
    frame = frame.groupby("date", as_index=False).mean().sort_values("date").reset_index(drop=True)
    return frame, paths


def load_reddit_daily_frame(
    raw_dir: Path,
    spec: dict[str, object],
    source_key: str,
) -> tuple[pd.DataFrame, list[Path]]:
    frame, paths = load_combined_csv_source(raw_dir, spec, source_key, "Reddit")
    frame = coerce_numeric_columns(frame, REDDIT_BASE_COLUMNS)
    frame = frame[["date"] + REDDIT_BASE_COLUMNS].sort_values("date").reset_index(drop=True)
    return frame, paths


def inspect_source(
    raw_dir: Path,
    spec: dict[str, object],
    source_name: str,
    candidates_key: str,
) -> dict[str, object]:
    candidates = tuple(spec[candidates_key])
    paths = existing_paths(raw_dir, candidates)
    if not paths:
        return {
            "ticker": spec["ticker"],
            "company_name": spec["company_name"],
            "source_name": source_name,
            "selected_files": None,
            "missing_candidates": ", ".join(candidates),
            "rows": 0,
            "date_min": None,
            "date_max": None,
        }

    frames = [read_csv_with_standard_date(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    return {
        "ticker": spec["ticker"],
        "company_name": spec["company_name"],
        "source_name": source_name,
        "selected_files": ", ".join(path.name for path in paths),
        "missing_candidates": ", ".join(
            candidate for candidate in candidates if not (raw_dir / candidate).exists()
        ),
        "rows": int(len(combined)),
        "date_min": combined["date"].min().date().isoformat() if not combined.empty else None,
        "date_max": combined["date"].max().date().isoformat() if not combined.empty else None,
    }


def build_raw_source_audit(raw_dir: Path, ticker_specs: list[dict[str, object]]) -> pd.DataFrame:
    records = []
    source_map = {
        "stock": "stock_candidates",
        "google_trends": "google_candidates",
        "reddit_submissions": "reddit_subm_candidates",
        "reddit_comments": "reddit_comm_candidates",
        "gdelt": "gdelt_candidates",
    }
    for spec in ticker_specs:
        for source_name, candidates_key in source_map.items():
            records.append(inspect_source(raw_dir, spec, source_name, candidates_key))
    return pd.DataFrame(records).sort_values(["ticker", "source_name"]).reset_index(drop=True)


def build_reddit_coverage_audit(
    raw_dir: Path,
    ticker_specs: list[dict[str, object]],
) -> pd.DataFrame:
    records = []
    for spec in ticker_specs:
        subm, subm_paths = load_reddit_daily_frame(raw_dir, spec, "reddit_subm_candidates")
        comm, comm_paths = load_reddit_daily_frame(raw_dir, spec, "reddit_comm_candidates")
        records.append(
            {
                "ticker": spec["ticker"],
                "company_name": spec["company_name"],
                "subm_source_files": ", ".join(path.name for path in subm_paths),
                "comm_source_files": ", ".join(path.name for path in comm_paths),
                "subm_days": int(len(subm)),
                "subm_days_with_posts": int((subm["reddit_posts"] > 0).sum()),
                "subm_post_coverage": float((subm["reddit_posts"] > 0).mean()),
                "subm_finbert_null_days": int(subm["reddit_finbert_mean"].isna().sum()),
                "subm_finbert_zero_days": int(subm["reddit_finbert_mean"].fillna(np.nan).eq(0).sum()),
                "comm_days": int(len(comm)),
                "comm_days_with_posts": int((comm["reddit_posts"] > 0).sum()),
                "comm_post_coverage": float((comm["reddit_posts"] > 0).mean()),
                "comm_finbert_null_days": int(comm["reddit_finbert_mean"].isna().sum()),
                "comm_finbert_zero_days": int(comm["reddit_finbert_mean"].fillna(np.nan).eq(0).sum()),
            }
        )
    return pd.DataFrame(records).sort_values("ticker").reset_index(drop=True)


def build_session_calendar(trading_dates: pd.Series) -> pd.DataFrame:
    return (
        pd.Series(pd.to_datetime(trading_dates).dropna().sort_values().unique(), name="session_date")
        .to_frame()
        .reset_index(drop=True)
    )


def map_calendar_to_next_session(frame: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    if frame.empty:
        working = frame.copy()
        working["session_date"] = pd.NaT
        return working

    session_calendar = build_session_calendar(trading_dates)
    working = frame.sort_values("date").copy()
    mapped = pd.merge_asof(
        working,
        session_calendar,
        left_on="date",
        right_on="session_date",
        direction="forward",
        allow_exact_matches=True,
    )
    return mapped.dropna(subset=["session_date"]).copy()


def align_google_to_sessions(google_df: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    mapped = map_calendar_to_next_session(google_df, trading_dates)
    aligned = mapped.groupby("session_date", as_index=False).agg(
        google_trends_score=("google_trends_score", "mean")
    )
    return aligned.rename(columns={"session_date": "date"})


def align_gdelt_to_sessions(gdelt_df: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    mapped = map_calendar_to_next_session(gdelt_df, trading_dates)
    aligned = mapped.groupby("session_date", as_index=False).agg(
        gdelt_articles=("gdelt_articles", "sum"),
        gdelt_robust=("gdelt_robust", "mean"),
        gdelt_sentiment_score=("gdelt_sentiment_score", "mean"),
    )
    return aligned.rename(columns={"session_date": "date"})


def prepare_reddit_for_merge(reddit_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename_map = {column: f"{prefix}_{column}" for column in REDDIT_BASE_COLUMNS}
    return reddit_df.rename(columns=rename_map)


def build_panel_for_ticker(raw_dir: Path, spec: dict[str, object]) -> pd.DataFrame:
    stock_df, _ = load_stock_frame(raw_dir, spec)
    google_df, _ = load_google_frame(raw_dir, spec)
    gdelt_df, _ = load_gdelt_frame(raw_dir, spec)
    subm_df, _ = load_reddit_daily_frame(raw_dir, spec, "reddit_subm_candidates")
    comm_df, _ = load_reddit_daily_frame(raw_dir, spec, "reddit_comm_candidates")

    panel = stock_df.copy()
    panel["ticker"] = spec["ticker"]
    panel["company_name"] = spec["company_name"]

    panel = panel.merge(prepare_reddit_for_merge(subm_df, "subm"), on="date", how="left")
    panel = panel.merge(prepare_reddit_for_merge(comm_df, "comm"), on="date", how="left")
    panel = panel.merge(align_google_to_sessions(google_df, stock_df["date"]), on="date", how="left")
    panel = panel.merge(align_gdelt_to_sessions(gdelt_df, stock_df["date"]), on="date", how="left")
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
        "output_path": str(output_dataset_path),
        "summary_path": str(output_summary_path),
        "source_audit_path": str(output_source_audit_path),
        "reddit_coverage_path": str(output_reddit_coverage_path),
        "notes": [
            "Raw source files are read-only inputs.",
            "Backfill files and current files are concatenated per ticker/source, then duplicate dates keep the later candidate.",
            "Older TSLA generic source files are used as fallbacks for the current period.",
            "Reddit daily files are merged with original source values.",
            "Google Trends is minimally mapped from calendar days to the next trading session using a mean.",
            "GDELT is minimally mapped from calendar days to the next trading session using sum for gdelt_articles and mean for gdelt_robust and gdelt_sentiment_score.",
            "No additional missingness flags or coverage features are added to the final panel.",
        ],
        "raw_source_rows": source_audit.to_dict(orient="records"),
        "coverage_rows": reddit_coverage_audit.to_dict(orient="records"),
    }


def build_full_history_dataset(
    *,
    project_root: Path | None = None,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    backfill_suffixes: tuple[str, ...] = DEFAULT_BACKFILL_SUFFIXES,
    output_prefix: str | None = None,
) -> dict[str, object]:
    root = find_project_root(project_root)
    raw_dir = root / "data" / "equity_data"
    datasets_dir = root / "data" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    resolved_tickers = normalize_tickers(tickers)
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
    reddit_coverage_audit = build_reddit_coverage_audit(raw_dir, ticker_specs)
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
    return normalize_tickers(tickers)


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
    result = build_full_history_dataset(
        tickers=tickers,
        backfill_suffixes=backfill_suffixes,
        output_prefix=args.output_prefix,
    )
    print(f"Saved panel to {result['output_dataset_path']}")
    print(f"Saved summary to {result['output_summary_path']}")
    print(f"Saved metadata to {result['output_metadata_path']}")
    return result


if __name__ == "__main__":
    main()
