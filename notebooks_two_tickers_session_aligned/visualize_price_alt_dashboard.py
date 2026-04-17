from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from price_alt_baseline_experiment import locate_panel_dataset


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
VISUAL_DIR = OUTPUT_DIR / "visualizations"


def _average_available(series_list: list[pd.Series]) -> pd.Series:
    valid = [series for series in series_list if series is not None]
    if not valid:
        return pd.Series(dtype="float64")
    if len(valid) == 1:
        return valid[0]
    return pd.concat(valid, axis=1).mean(axis=1)


def _rolling_zscore(series: pd.Series, window: int = 20, min_periods: int = 10) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0.0, np.nan)


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def load_panel(dataset_path: Path | None = None) -> pd.DataFrame:
    dataset_path = dataset_path or locate_panel_dataset()
    panel = pd.read_csv(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    for column in panel.columns:
        if column not in {"date", "ticker", "company_name", "y"}:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")

    pieces: list[pd.DataFrame] = []
    for _, ticker_frame in panel.groupby("ticker", sort=True):
        work = ticker_frame.sort_values("date").reset_index(drop=True).copy()

        subm_posts = _safe_numeric(work, "subm_reddit_posts").fillna(0.0)
        comm_posts = _safe_numeric(work, "comm_reddit_posts").fillna(0.0)
        subm_comments = _safe_numeric(work, "subm_reddit_comments_sum").fillna(0.0)
        comm_comments = _safe_numeric(work, "comm_reddit_comments_sum").fillna(0.0)

        subm_vader = _safe_numeric(work, "subm_reddit_vader_weighted_mean")
        comm_vader = _safe_numeric(work, "comm_reddit_vader_weighted_mean")
        subm_finbert = _safe_numeric(work, "subm_reddit_finbert_weighted_mean")
        comm_finbert = _safe_numeric(work, "comm_reddit_finbert_weighted_mean")

        subm_direction = _average_available([subm_vader, subm_finbert])
        comm_direction = _average_available([comm_vader, comm_finbert])
        reddit_combined_direction = _average_available([subm_direction, comm_direction])

        reddit_posts_total = subm_posts + comm_posts
        reddit_comments_total = subm_comments + comm_comments

        work["subm_direction"] = subm_direction
        work["comm_direction"] = comm_direction
        work["reddit_combined_direction"] = reddit_combined_direction
        work["reddit_posts_total"] = reddit_posts_total
        work["reddit_comments_total"] = reddit_comments_total
        work["price_base_100"] = 100.0 * work["stock_price"] / work["stock_price"].iloc[0]

        work["price_z20"] = _rolling_zscore(np.log(work["stock_price"]), 20)
        work["trends_z20"] = _rolling_zscore(_safe_numeric(work, "google_trends_score"), 20)
        work["gdelt_articles_z20"] = _rolling_zscore(
            np.log1p(_safe_numeric(work, "gdelt_articles").clip(lower=0.0)),
            20,
        )
        work["reddit_attention_z20"] = _rolling_zscore(np.log1p(reddit_posts_total.clip(lower=0.0)), 20)
        work["reddit_sentiment_z20"] = _rolling_zscore(reddit_combined_direction, 20)

        pieces.append(work)

    return pd.concat(pieces, ignore_index=True)


def filter_panel(
    panel: pd.DataFrame,
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    view = panel[panel["ticker"] == ticker].copy()
    if start is not None:
        view = view[view["date"] >= pd.Timestamp(start)]
    if end is not None:
        view = view[view["date"] <= pd.Timestamp(end)]
    return view.sort_values("date").reset_index(drop=True)


def summarize_window(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": frame["ticker"].iloc[0],
                "rows": int(len(frame)),
                "date_min": frame["date"].min().date().isoformat(),
                "date_max": frame["date"].max().date().isoformat(),
                "future_up_share": float((frame["future_return_1d"] > 0).mean()),
                "avg_price_return_1d": float(frame["stock_return_1d"].mean()),
                "avg_trends": float(frame["google_trends_score"].mean()),
                "avg_gdelt_articles": float(frame["gdelt_articles"].mean()),
                "avg_reddit_posts_total": float(frame["reddit_posts_total"].mean()),
            }
        ]
    )


def _format_time_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", rotation=45)


def plot_ticker_dashboard(
    panel: pd.DataFrame,
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    figsize: tuple[int, int] = (16, 20),
) -> tuple[plt.Figure, np.ndarray]:
    frame = filter_panel(panel, ticker=ticker, start=start, end=end)
    if frame.empty:
        raise ValueError(f"No rows available for ticker={ticker}, start={start}, end={end}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        nrows=6,
        ncols=1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )

    fig.suptitle(
        f"{ticker}: Stock Price vs Alternative Data\n{frame['date'].min().date()} - {frame['date'].max().date()}",
        fontsize=15,
        y=1.02,
    )

    axes[0].plot(frame["date"], frame["stock_price"], color="#1f4e79", linewidth=2.0, label="stock_price")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left")
    volume_axis = axes[0].twinx()
    volume_axis.bar(
        frame["date"],
        frame["stock_volume"],
        width=1.8,
        alpha=0.18,
        color="#7f8c8d",
        label="stock_volume",
    )
    volume_axis.set_ylabel("Volume", color="#7f8c8d")
    volume_axis.tick_params(axis="y", colors="#7f8c8d")

    axes[1].plot(frame["date"], frame["google_trends_score"], color="#d35400", linewidth=1.8, label="google_trends")
    axes[1].set_ylabel("Trends")
    axes[1].legend(loc="upper left")
    gdelt_articles_axis = axes[1].twinx()
    gdelt_articles_axis.plot(
        frame["date"],
        frame["gdelt_articles"],
        color="#8e44ad",
        linewidth=1.2,
        alpha=0.9,
        label="gdelt_articles",
    )
    gdelt_articles_axis.set_ylabel("GDELT articles", color="#8e44ad")
    gdelt_articles_axis.tick_params(axis="y", colors="#8e44ad")

    axes[2].plot(
        frame["date"],
        frame["gdelt_sentiment_score"],
        color="#16a085",
        linewidth=1.6,
        label="gdelt_sentiment_score",
    )
    axes[2].plot(
        frame["date"],
        frame["gdelt_robust"],
        color="#2c3e50",
        linewidth=1.2,
        alpha=0.8,
        label="gdelt_robust",
    )
    axes[2].axhline(0.0, color="#95a5a6", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("GDELT sentiment")
    axes[2].legend(loc="upper left", ncol=2)

    axes[3].plot(
        frame["date"],
        frame["subm_reddit_posts"],
        color="#c0392b",
        linewidth=1.3,
        label="subm_reddit_posts",
    )
    axes[3].plot(
        frame["date"],
        frame["comm_reddit_posts"],
        color="#2980b9",
        linewidth=1.3,
        label="comm_reddit_posts",
    )
    axes[3].plot(
        frame["date"],
        frame["reddit_posts_total"],
        color="#34495e",
        linewidth=1.8,
        alpha=0.9,
        label="reddit_posts_total",
    )
    axes[3].set_ylabel("Reddit activity")
    axes[3].legend(loc="upper left", ncol=3)

    axes[4].plot(
        frame["date"],
        frame["subm_direction"],
        color="#e67e22",
        linewidth=1.3,
        label="subm_direction",
    )
    axes[4].plot(
        frame["date"],
        frame["comm_direction"],
        color="#27ae60",
        linewidth=1.3,
        label="comm_direction",
    )
    axes[4].plot(
        frame["date"],
        frame["reddit_combined_direction"],
        color="#8e44ad",
        linewidth=2.0,
        label="reddit_combined_direction",
    )
    axes[4].axhline(0.0, color="#95a5a6", linewidth=0.8, linestyle="--")
    axes[4].set_ylabel("Reddit sentiment")
    axes[4].legend(loc="upper left", ncol=3)

    overlay_cols = [
        ("price_z20", "price_z20", "#1f4e79"),
        ("trends_z20", "trends_z20", "#d35400"),
        ("gdelt_articles_z20", "gdelt_articles_z20", "#8e44ad"),
        ("reddit_attention_z20", "reddit_attention_z20", "#34495e"),
        ("reddit_sentiment_z20", "reddit_sentiment_z20", "#27ae60"),
    ]
    for col, label, color in overlay_cols:
        axes[5].plot(frame["date"], frame[col], linewidth=1.5, color=color, label=label)
    axes[5].axhline(0.0, color="#95a5a6", linewidth=0.8, linestyle="--")
    axes[5].set_ylabel("z-score")
    axes[5].set_title("Normalized overlay for visual pattern check")
    axes[5].legend(loc="upper left", ncol=3)

    for axis in axes:
        _format_time_axis(axis)

    axes[-1].set_xlabel("Date")
    return fig, axes


def plot_normalized_overlay(
    panel: pd.DataFrame,
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    figsize: tuple[int, int] = (16, 6),
) -> tuple[plt.Figure, plt.Axes]:
    frame = filter_panel(panel, ticker=ticker, start=start, end=end)
    if frame.empty:
        raise ValueError(f"No rows available for ticker={ticker}, start={start}, end={end}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    series_defs = [
        ("price_z20", "#1f4e79"),
        ("trends_z20", "#d35400"),
        ("gdelt_articles_z20", "#8e44ad"),
        ("reddit_attention_z20", "#34495e"),
        ("reddit_sentiment_z20", "#27ae60"),
    ]
    for col, color in series_defs:
        axis.plot(frame["date"], frame[col], linewidth=1.6, label=col, color=color)
    axis.axhline(0.0, color="#95a5a6", linewidth=0.8, linestyle="--")
    axis.set_title(f"{ticker}: normalized overlay")
    axis.set_ylabel("z-score")
    _format_time_axis(axis)
    axis.legend(loc="upper left", ncol=3)
    return fig, axis


def plot_price_vs_reddit_sentiment(
    panel: pd.DataFrame,
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    sentiment_variant: str = "combined",
    normalize_price: bool = False,
    figsize: tuple[int, int] = (16, 6),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    frame = filter_panel(panel, ticker=ticker, start=start, end=end)
    if frame.empty:
        raise ValueError(f"No rows available for ticker={ticker}, start={start}, end={end}")

    sentiment_map = {
        "combined": ("reddit_combined_direction", "reddit_combined_direction"),
        "subm": ("subm_direction", "subm_direction"),
        "comm": ("comm_direction", "comm_direction"),
    }
    if sentiment_variant not in sentiment_map:
        raise ValueError(
            "sentiment_variant must be one of: 'combined', 'subm', 'comm'"
        )

    sentiment_col, sentiment_label = sentiment_map[sentiment_variant]
    price_col = "price_base_100" if normalize_price else "stock_price"
    price_label = "price_base_100" if normalize_price else "stock_price"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis_left = plt.subplots(figsize=figsize, constrained_layout=True)
    axis_right = axis_left.twinx()

    axis_left.plot(
        frame["date"],
        frame[price_col],
        color="#1f4e79",
        linewidth=2.0,
        label=price_label,
    )
    axis_left.set_ylabel("Price" if not normalize_price else "Price (base=100)", color="#1f4e79")
    axis_left.tick_params(axis="y", colors="#1f4e79")

    axis_right.plot(
        frame["date"],
        frame[sentiment_col],
        color="#c0392b",
        linewidth=1.6,
        alpha=0.9,
        label=sentiment_label,
    )
    axis_right.axhline(0.0, color="#95a5a6", linewidth=0.8, linestyle="--")
    axis_right.set_ylabel("Reddit sentiment", color="#c0392b")
    axis_right.tick_params(axis="y", colors="#c0392b")

    _format_time_axis(axis_left)
    axis_left.set_title(
        f"{ticker}: stock price vs Reddit sentiment ({sentiment_variant})\n"
        f"{frame['date'].min().date()} - {frame['date'].max().date()}"
    )

    left_handles, left_labels = axis_left.get_legend_handles_labels()
    right_handles, right_labels = axis_right.get_legend_handles_labels()
    axis_left.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left")
    return fig, (axis_left, axis_right)


def save_dashboard(
    panel: pd.DataFrame,
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    filename: str | None = None,
) -> Path:
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    frame = filter_panel(panel, ticker=ticker, start=start, end=end)
    if frame.empty:
        raise ValueError(f"No rows available for ticker={ticker}, start={start}, end={end}")

    safe_start = frame["date"].min().date().isoformat()
    safe_end = frame["date"].max().date().isoformat()
    filename = filename or f"{ticker.lower()}_{safe_start}_{safe_end}_dashboard.png"
    output_path = VISUAL_DIR / filename

    fig, _ = plot_ticker_dashboard(panel, ticker=ticker, start=start, end=end)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
