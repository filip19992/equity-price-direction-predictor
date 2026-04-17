import argparse
import datetime as dt
from pathlib import Path

from equity_data_importers.config import (
    TICKER_GROUPS,
    Config,
    build_profiled_config,
    get_group_tickers,
    parse_ticker_values,
)
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


def validate_date_range(start_date: dt.date, end_date: dt.date) -> None:
    if end_date < start_date:
        raise argparse.ArgumentTypeError(
            f"Invalid date range: end_date={end_date} is earlier than start_date={start_date}."
        )


def build_config(args: argparse.Namespace) -> Config:
    defaults = Config()
    ticker = (args.ticker or defaults.TICKER).strip().upper()
    start_date = args.start_date or defaults.START_DATE
    end_date = args.end_date or defaults.END_DATE
    validate_date_range(start_date, end_date)

    return build_profiled_config(
        ticker=ticker,
        company_name=args.company_name,
        trends_query=args.trends_query,
        gdelt_query=args.gdelt_query,
        geo=args.geo or defaults.GEO,
        start_date=start_date,
        end_date=end_date,
        finbert_required=(
            defaults.FINBERT_REQUIRED if args.finbert_required is None else args.finbert_required
        ),
        reddit_submissions_source=(
            args.reddit_submissions_source or defaults.REDDIT_SUBMISSIONS_SOURCE
        ),
        reddit_comments_source=(
            args.reddit_comments_source or defaults.REDDIT_COMMENTS_SOURCE
        ),
        output_tag=args.output_tag,
    )


def get_expected_output_paths(importer: object) -> tuple[Path, ...]:
    if isinstance(importer, GoogleTrendsImporter):
        return (
            importer.output_path(
                legacy_name="google_trends_data.csv",
                generic_stem="google_trends_data",
            ),
        )
    if isinstance(importer, GdeltImporter):
        return (
            importer.output_path(
                legacy_name="gdelt_data.csv",
                generic_stem="gdelt_data",
            ),
        )
    if isinstance(importer, RedditImporter):
        return (importer.raw_output_path, importer.daily_output_path)
    if isinstance(importer, RedditCommentsImporter):
        return (importer.raw_output_path, importer.daily_output_path)
    if isinstance(importer, StockPriceImporter):
        return (
            importer.output_path(
                legacy_name="stock-prices-data.csv",
                generic_stem="stock-prices-data",
            ),
        )
    return ()


def run_importers(
    selected: list[str] | None = None,
    config: Config | None = None,
    skip_existing: bool = False,
    continue_on_error: bool = False,
) -> dict[str, object]:
    runtime_config = config or Config()
    importer_names = selected or DEFAULT_IMPORTERS
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

    for importer_name in importer_names:
        importer = IMPORTERS[importer_name](config=runtime_config)
        expected_outputs = get_expected_output_paths(importer)
        if skip_existing and expected_outputs and all(path.exists() for path in expected_outputs):
            print(
                f"Skipping importer: {importer.name} "
                f"(existing outputs: {', '.join(str(path) for path in expected_outputs)})"
            )
            results[importer.name] = (
                expected_outputs[0] if len(expected_outputs) == 1 else expected_outputs
            )
            continue

        print(f"Running importer: {importer.name}")
        try:
            results[importer.name] = importer.run()
        except Exception as exc:
            if not continue_on_error:
                raise
            print(f"Importer failed: {importer.name} ({exc})")
            results[importer.name] = {"error": str(exc)}

    return results


def collect_requested_tickers(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    if args.ticker_group:
        requested.extend(get_group_tickers(args.ticker_group))
    if args.tickers:
        requested.extend(parse_ticker_values(args.tickers))
    if args.ticker:
        requested.extend(parse_ticker_values([args.ticker]))

    seen: set[str] = set()
    deduplicated: list[str] = []
    for ticker in requested:
        if ticker in seen:
            continue
        seen.add(ticker)
        deduplicated.append(ticker)

    excluded = set(parse_ticker_values(args.exclude_tickers))
    return [ticker for ticker in deduplicated if ticker not in excluded]


def build_configs(args: argparse.Namespace) -> list[Config]:
    defaults = Config()
    start_date = args.start_date or defaults.START_DATE
    end_date = args.end_date or defaults.END_DATE
    validate_date_range(start_date, end_date)

    tickers = collect_requested_tickers(args)
    if not tickers:
        return [build_config(args)]

    if len(tickers) > 1:
        restricted_args = {
            "company_name": args.company_name,
            "trends_query": args.trends_query,
            "gdelt_query": args.gdelt_query,
            "output_tag": args.output_tag,
        }
        active_restricted = sorted(
            name for name, value in restricted_args.items() if value is not None
        )
        if active_restricted:
            raise argparse.ArgumentTypeError(
                "These arguments are only allowed for a single ticker run: "
                + ", ".join(active_restricted)
            )

    return [
        build_profiled_config(
            ticker=ticker,
            geo=args.geo or defaults.GEO,
            start_date=start_date,
            end_date=end_date,
            finbert_required=(
                defaults.FINBERT_REQUIRED
                if args.finbert_required is None
                else args.finbert_required
            ),
            reddit_submissions_source=(
                args.reddit_submissions_source or defaults.REDDIT_SUBMISSIONS_SOURCE
            ),
            reddit_comments_source=(
                args.reddit_comments_source or defaults.REDDIT_COMMENTS_SOURCE
            ),
        )
        for ticker in tickers
    ]


def run_batch(
    configs: list[Config],
    selected: list[str] | None = None,
    skip_existing: bool = False,
    continue_on_error: bool = False,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for index, config in enumerate(configs, start=1):
        print(f"\n=== [{index}/{len(configs)}] {config.TICKER} / {config.COMPANY_NAME} ===")
        results[config.TICKER] = run_importers(
            selected=selected,
            config=config,
            skip_existing=skip_existing,
            continue_on_error=continue_on_error,
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "importers",
        nargs="*",
        metavar="IMPORTER",
        help="Optional importer names to run. If omitted, all importers run.",
    )
    parser.add_argument("--ticker", help="Single stock ticker, e.g. TSLA, AAPL, MSFT.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Batch of tickers to run. Accepts whitespace-separated or comma-separated values.",
    )
    parser.add_argument(
        "--ticker-group",
        choices=sorted(TICKER_GROUPS),
        help="Predefined ticker batch to run.",
    )
    parser.add_argument(
        "--exclude-tickers",
        nargs="+",
        help="Tickers to remove from --tickers/--ticker-group, e.g. AAPL TSLA.",
    )
    parser.add_argument(
        "--company-name",
        help="Company name used by keyword matching and default Trends query.",
    )
    parser.add_argument(
        "--trends-query",
        help="Override Google Trends query. Defaults to company profile or company name.",
    )
    parser.add_argument(
        "--gdelt-query",
        help="Override GDELT query. Defaults to company profile or '(Company OR TICKER)'.",
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
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip importers whose expected output files already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue batch execution if a single importer fails for one ticker.",
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


def main() -> dict[str, object] | dict[str, dict[str, object]]:
    args = parse_args()
    try:
        configs = build_configs(args)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc

    if len(configs) == 1:
        return run_importers(
            args.importers,
            config=configs[0],
            skip_existing=args.skip_existing,
            continue_on_error=args.continue_on_error,
        )
    return run_batch(
        configs,
        selected=args.importers,
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()
