import argparse
import datetime as dt

from equity_data_importers.config import Config
from equity_data_importers.importers import (
    GdeltImporter,
    GoogleTrendsImporter,
    RedditCommentsImporter,
    RedditImporter,
    StockPriceImporter,
)

IMPORTERS = {
    "google_trends": GoogleTrendsImporter,
    "gdelt": GdeltImporter,
    "reddit": RedditImporter,
    "reddit_comments": RedditCommentsImporter,
    "stock_price": StockPriceImporter,
}

DEFAULT_IMPORTERS = [
    "google_trends",
    "gdelt",
    "reddit",
    "reddit_comments",
    "stock_price",
]


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected format YYYY-MM-DD."
        ) from exc


def build_config(args: argparse.Namespace) -> Config:
    defaults = Config()

    ticker = (args.ticker or defaults.TICKER).strip().upper()
    if args.company_name is not None:
        company_name = args.company_name.strip()
    elif args.ticker is not None:
        company_name = ticker
    else:
        company_name = defaults.COMPANY_NAME

    start_date = args.start_date or defaults.START_DATE
    end_date = args.end_date or defaults.END_DATE
    if end_date < start_date:
        raise argparse.ArgumentTypeError(
            f"Invalid date range: end_date={end_date} is earlier than start_date={start_date}."
        )

    return Config(
        COMPANY_NAME=company_name,
        TICKER=ticker,
        GDELT_QUERY=args.gdelt_query,
        TRENDS_QUERY=args.trends_query,
        GEO=(args.geo or defaults.GEO).strip(),
        START_DATE=start_date,
        END_DATE=end_date,
        FINBERT_REQUIRED=(
            defaults.FINBERT_REQUIRED
            if args.finbert_required is None
            else args.finbert_required
        ),
        REDDIT_SUBMISSIONS_SOURCE=(
            args.reddit_submissions_source or defaults.REDDIT_SUBMISSIONS_SOURCE
        ),
        REDDIT_COMMENTS_SOURCE=(
            args.reddit_comments_source or defaults.REDDIT_COMMENTS_SOURCE
        ),
        OUTPUT_TAG=args.output_tag,
    )


def run_importers(
    selected: list[str] | None = None,
    config: Config | None = None,
) -> dict[str, object]:
    runtime_config = config or Config()
    importer_names = selected or DEFAULT_IMPORTERS
    importers = [IMPORTERS[name](config=runtime_config) for name in importer_names]
    results: dict[str, object] = {}

    print(
        "Runtime config: "
        f"ticker={runtime_config.TICKER}, "
        f"company={runtime_config.COMPANY_NAME}, "
        f"geo={runtime_config.GEO}, "
        f"start={runtime_config.START_DATE}, "
        f"end={runtime_config.END_DATE}, "
        f"output_tag={runtime_config.resolved_output_tag}"
    )

    for importer in importers:
        print(f"Running importer: {importer.name}")
        results[importer.name] = importer.run()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "importers",
        nargs="*",
        metavar="IMPORTER",
        help="Optional importer names to run. If omitted, all importers run.",
    )
    parser.add_argument("--ticker", help="Stock ticker, e.g. TSLA, AAPL, MSFT.")
    parser.add_argument(
        "--company-name",
        help="Company name used by keyword matching and default Trends query.",
    )
    parser.add_argument(
        "--trends-query",
        help="Override Google Trends query. Defaults to company name.",
    )
    parser.add_argument(
        "--gdelt-query",
        help="Override GDELT query. Defaults to '(Company OR TICKER)'.",
    )
    parser.add_argument("--geo", help="Geo code used for Google Trends/GDELT, e.g. US.")
    parser.add_argument(
        "--start-date",
        type=parse_date,
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-tag",
        help="Optional output suffix/tag for generated files. Defaults to ticker.",
    )
    parser.add_argument(
        "--reddit-submissions-source",
        help="Input filename for reddit submissions NDJSON (in data/equity_data).",
    )
    parser.add_argument(
        "--reddit-comments-source",
        help="Input filename for reddit comments NDJSON (in data/equity_data).",
    )
    parser.add_argument(
        "--finbert-required",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require FinBERT scoring (default: true).",
    )
    args = parser.parse_args()
    invalid = sorted({name for name in args.importers if name not in IMPORTERS})
    if invalid:
        parser.error(
            "argument importers: invalid choice(s): "
            + ", ".join(invalid)
            + " (choose from "
            + ", ".join(sorted(IMPORTERS.keys()))
            + ")"
        )
    return args


def run_all() -> dict[str, object]:
    return run_importers()


def main() -> dict[str, object]:
    args = parse_args()
    try:
        config = build_config(args)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    return run_importers(args.importers, config=config)


if __name__ == "__main__":
    main()
