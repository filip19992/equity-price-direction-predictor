from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "equity_data"
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
DATASET_DIR = OUTPUT_DIR / "datasets"
PROCESSED_SOURCES_DIR = OUTPUT_DIR / "processed_sources"


@dataclass(frozen=True)
class TickerProfile:
    ticker: str
    company_name: str
    output_tag: str


DEFAULT_PROFILES = [
    TickerProfile(ticker="TSLA", company_name="Tesla", output_tag="tsla"),
    TickerProfile(ticker="AAPL", company_name="Apple", output_tag="aapl"),
]

Y_CODE_MAP = {"spadek": 0, "bez_zmian": 1, "wzrost": 2}


def _candidate_names(prefix: str, profile: TickerProfile) -> list[str]:
    tagged = f"{prefix}_{profile.output_tag}.csv"
    if profile.output_tag == "tsla":
        return [tagged, f"{prefix}.csv"]
    return [tagged]


def locate_existing_file(base_dir: Path, candidates: list[str], label: str) -> Path:
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing {label}. Checked: {', '.join(str(base_dir / name) for name in candidates)}"
    )


def read_daily_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_col = "date" if "date" in frame.columns else "Date" if "Date" in frame.columns else None
    if date_col is None:
        raise ValueError(f"File {path} does not contain a date column.")

    if date_col != "date":
        frame = frame.rename(columns={date_col: "date"})

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"]).copy()

    for col in frame.columns:
        if col != "date":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame.sort_values("date").reset_index(drop=True)


def prefix_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename_map = {col: f"{prefix}{col}" for col in frame.columns if col != "date"}
    return frame.rename(columns=rename_map)


def collapse_gdelt_to_daily(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    daily = (
        working.groupby("date", as_index=False)
        .mean(numeric_only=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


def fill_daily_gdelt_range(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    full_range = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    daily_full = frame.set_index("date").reindex(full_range)
    daily_full.index.name = "date"
    daily_full = daily_full.interpolate(method="time", limit_direction="both")
    return daily_full.reset_index()


def map_to_trading_sessions(frame: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    trading_calendar = pd.DataFrame({"session_date": pd.Series(trading_dates).drop_duplicates().sort_values()})
    mapped = pd.merge_asof(
        frame.sort_values("date"),
        trading_calendar,
        left_on="date",
        right_on="session_date",
        direction="forward",
    )
    mapped = mapped.dropna(subset=["session_date"]).copy()
    mapped = mapped.rename(columns={"date": "source_date", "session_date": "date"})
    return mapped.reset_index(drop=True)


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").clip(lower=0)
    valid = values.notna() & weights.notna()
    if not valid.any():
        return np.nan
    total_weight = float(weights.loc[valid].sum())
    if total_weight <= 0:
        return float(values.loc[valid].mean())
    return float((values.loc[valid] * weights.loc[valid]).sum() / total_weight)


def aggregate_reddit_to_sessions(frame: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    mapped = map_to_trading_sessions(frame, trading_dates)
    group = mapped.groupby("date", sort=True)

    sum_cols = [
        col
        for col in mapped.columns
        if col != "date" and col != "source_date" and (col == "reddit_posts" or col.endswith("_sum"))
    ]
    mean_cols = [col for col in mapped.columns if col.endswith("_std")]

    aggregated = group[sum_cols].sum(min_count=1)
    if mean_cols:
        aggregated = aggregated.join(group[mean_cols].mean())

    if "reddit_vader_sum" in aggregated.columns and "reddit_posts" in aggregated.columns:
        aggregated["reddit_vader_mean"] = aggregated["reddit_vader_sum"] / aggregated["reddit_posts"].replace(0, np.nan)
    if "reddit_finbert_sum" in aggregated.columns and "reddit_posts" in aggregated.columns:
        aggregated["reddit_finbert_mean"] = aggregated["reddit_finbert_sum"] / aggregated["reddit_posts"].replace(0, np.nan)
    if "reddit_sent_sum" in aggregated.columns and "reddit_posts" in aggregated.columns:
        aggregated["reddit_sent_mean"] = aggregated["reddit_sent_sum"] / aggregated["reddit_posts"].replace(0, np.nan)

    if {"reddit_vader_weighted_mean", "reddit_weight_sum"}.issubset(mapped.columns):
        aggregated["reddit_vader_weighted_mean"] = group.apply(
            lambda g: weighted_average(g["reddit_vader_weighted_mean"], g["reddit_weight_sum"])
        )
    if {"reddit_finbert_weighted_mean", "reddit_weight_sum"}.issubset(mapped.columns):
        aggregated["reddit_finbert_weighted_mean"] = group.apply(
            lambda g: weighted_average(g["reddit_finbert_weighted_mean"], g["reddit_weight_sum"])
        )

    output_cols = [
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
    output = aggregated.reset_index()
    output = output[[col for col in ["date"] + output_cols if col in output.columns]]
    return output.sort_values("date").reset_index(drop=True)


def aggregate_trends_to_sessions(frame: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    mapped = map_to_trading_sessions(frame, trading_dates)
    aggregated = (
        mapped.groupby("date", as_index=False)["trends_score"]
        .mean()
        .rename(columns={"trends_score": "google_trends_score"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return aggregated


def aggregate_gdelt_to_sessions(frame: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    mapped = map_to_trading_sessions(frame, trading_dates)
    group = mapped.groupby("date", sort=True)
    aggregated = group["gdelt_articles"].sum(min_count=1).to_frame()

    aggregated["gdelt_robust"] = group.apply(
        lambda g: weighted_average(g["gdelt_robust"], g["gdelt_articles"])
        if "gdelt_articles" in g.columns and "gdelt_robust" in g.columns
        else np.nan
    )
    aggregated["gdelt_sentiment_score"] = group.apply(
        lambda g: weighted_average(g["gdelt_sentiment_score"], g["gdelt_articles"])
        if "gdelt_articles" in g.columns and "gdelt_sentiment_score" in g.columns
        else np.nan
    )

    output = aggregated.reset_index()
    return output.sort_values("date").reset_index(drop=True)


def prepare_gdelt_frame(profile: TickerProfile, data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict[str, str]]:
    candidates = []
    for stem in ("gdelt_data_daily_filled", "gdelt_data_daily", "gdelt_data"):
        candidates.extend(_candidate_names(stem, profile))

    source_path = locate_existing_file(data_dir, candidates, f"GDELT source for {profile.ticker}")
    raw = read_daily_csv(source_path)
    daily = collapse_gdelt_to_daily(raw)
    daily_filled = fill_daily_gdelt_range(daily)

    if "sentiment_score" in daily_filled.columns:
        daily_filled = daily_filled.rename(columns={"sentiment_score": "gdelt_sentiment_score"})
    if "sentiment_score" in daily.columns:
        daily = daily.rename(columns={"sentiment_score": "gdelt_sentiment_score"})

    meta = {
        "source_path": str(source_path),
        "source_rows": str(len(raw)),
        "daily_rows": str(len(daily)),
        "filled_rows": str(len(daily_filled)),
    }
    return daily_filled, meta


def build_session_aligned_dataset_for_ticker(
    profile: TickerProfile,
    data_dir: Path = DATA_DIR,
    neutral_band: float = 0.002,
) -> tuple[pd.DataFrame, dict[str, object]]:
    stock_path = locate_existing_file(
        data_dir,
        _candidate_names("stock-prices-data", profile),
        f"stock prices for {profile.ticker}",
    )
    trends_path = locate_existing_file(
        data_dir,
        _candidate_names("google_trends_data", profile),
        f"google trends for {profile.ticker}",
    )
    reddit_subm_path = locate_existing_file(
        data_dir,
        _candidate_names("stock-reddit-data", profile),
        f"reddit submissions for {profile.ticker}",
    )
    reddit_comm_path = locate_existing_file(
        data_dir,
        _candidate_names("stock-reddit-comments-data", profile),
        f"reddit comments for {profile.ticker}",
    )

    gdelt_daily, gdelt_meta = prepare_gdelt_frame(profile=profile, data_dir=data_dir)

    stock_df = read_daily_csv(stock_path)
    trading_dates = stock_df["date"]

    reddit_subm_df = aggregate_reddit_to_sessions(read_daily_csv(reddit_subm_path), trading_dates)
    reddit_comm_df = aggregate_reddit_to_sessions(read_daily_csv(reddit_comm_path), trading_dates)
    trends_df = aggregate_trends_to_sessions(read_daily_csv(trends_path), trading_dates)
    gdelt_df = aggregate_gdelt_to_sessions(gdelt_daily, trading_dates)

    PROCESSED_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    reddit_subm_aligned_path = PROCESSED_SOURCES_DIR / f"reddit_submissions_session_aligned_{profile.output_tag}.csv"
    reddit_comm_aligned_path = PROCESSED_SOURCES_DIR / f"reddit_comments_session_aligned_{profile.output_tag}.csv"
    trends_aligned_path = PROCESSED_SOURCES_DIR / f"google_trends_session_aligned_{profile.output_tag}.csv"
    gdelt_aligned_path = PROCESSED_SOURCES_DIR / f"gdelt_session_aligned_{profile.output_tag}.csv"

    reddit_subm_df.to_csv(reddit_subm_aligned_path, index=False)
    reddit_comm_df.to_csv(reddit_comm_aligned_path, index=False)
    trends_df.to_csv(trends_aligned_path, index=False)
    gdelt_df.to_csv(gdelt_aligned_path, index=False)

    dataset = (
        stock_df.merge(prefix_columns(reddit_subm_df, "subm_"), on="date", how="left")
        .merge(prefix_columns(reddit_comm_df, "comm_"), on="date", how="left")
        .merge(trends_df, on="date", how="left")
        .merge(gdelt_df, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    dataset["future_return_1d"] = dataset["stock_price"].shift(-1) / dataset["stock_price"] - 1
    dataset["y"] = np.select(
        [
            dataset["future_return_1d"] < -neutral_band,
            dataset["future_return_1d"].abs() <= neutral_band,
            dataset["future_return_1d"] > neutral_band,
        ],
        ["spadek", "bez_zmian", "wzrost"],
        default=pd.NA,
    )
    dataset["y_code"] = dataset["y"].map(Y_CODE_MAP).astype("float64")
    dataset["stock_return_1d"] = dataset["stock_price"].pct_change(1)
    dataset["stock_return_5d"] = dataset["stock_price"].pct_change(5)
    dataset["stock_volume_change_1d"] = dataset["stock_volume"].pct_change(1)
    dataset["weekday"] = dataset["date"].dt.dayofweek.astype(float)
    dataset["ticker"] = profile.ticker
    dataset["company_name"] = profile.company_name

    ordered_front = [
        "date",
        "ticker",
        "company_name",
        "y",
        "y_code",
        "future_return_1d",
        "stock_price",
        "stock_volume",
    ]
    other_cols = [col for col in dataset.columns if col not in ordered_front]
    dataset = dataset[ordered_front + other_cols]

    meta = {
        "profile": asdict(profile),
        "source_paths": {
            "stock": str(stock_path),
            "google_trends": str(trends_path),
            "reddit_submissions": str(reddit_subm_path),
            "reddit_comments": str(reddit_comm_path),
            "gdelt": gdelt_meta,
        },
        "processed_source_paths": {
            "reddit_submissions_session_aligned": str(reddit_subm_aligned_path),
            "reddit_comments_session_aligned": str(reddit_comm_aligned_path),
            "google_trends_session_aligned": str(trends_aligned_path),
            "gdelt_session_aligned": str(gdelt_aligned_path),
        },
        "rows": int(len(dataset)),
        "date_min": dataset["date"].min().date().isoformat(),
        "date_max": dataset["date"].max().date().isoformat(),
        "neutral_band": neutral_band,
        "alignment_policy": "calendar-day alternative data mapped to same or next trading session",
        "columns": list(dataset.columns),
    }
    return dataset, meta


def build_two_ticker_session_aligned_dataset(
    profiles: list[TickerProfile] | None = None,
    data_dir: Path = DATA_DIR,
    dataset_dir: Path = DATASET_DIR,
    output_name: str = "stock_direction_two_tickers_session_aligned_base.csv",
    neutral_band: float = 0.002,
) -> dict[str, object]:
    profiles = profiles or list(DEFAULT_PROFILES)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    metadata = []
    for profile in profiles:
        dataset, meta = build_session_aligned_dataset_for_ticker(
            profile=profile,
            data_dir=data_dir,
            neutral_band=neutral_band,
        )
        datasets.append(dataset)
        metadata.append(meta)

    panel = pd.concat(datasets, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    output_path = dataset_dir / output_name
    panel.to_csv(output_path, index=False)

    summary = (
        panel.groupby("ticker", as_index=False)
        .agg(
            rows=("date", "size"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            n_positive=("y", lambda s: int((s == "wzrost").sum())),
            n_negative=("y", lambda s: int((s == "spadek").sum())),
            n_neutral=("y", lambda s: int((s == "bez_zmian").sum())),
        )
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    summary["date_min"] = summary["date_min"].dt.date.astype(str)
    summary["date_max"] = summary["date_max"].dt.date.astype(str)

    summary_path = dataset_dir / "stock_direction_two_tickers_session_aligned_summary.csv"
    summary.to_csv(summary_path, index=False)

    metadata_path = dataset_dir / "stock_direction_two_tickers_session_aligned_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "output_path": str(output_path),
                "summary_path": str(summary_path),
                "neutral_band": neutral_band,
                "profiles": [asdict(profile) for profile in profiles],
                "alignment_policy": "calendar-day alternative data mapped to same or next trading session",
                "per_ticker": metadata,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    return {
        "output_path": output_path,
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "panel": panel,
        "summary": summary,
        "metadata": metadata,
    }


def main() -> None:
    result = build_two_ticker_session_aligned_dataset()
    print(f"Saved panel dataset to: {result['output_path']}")
    print(f"Saved summary to: {result['summary_path']}")
    print(f"Saved metadata to: {result['metadata_path']}")
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
