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


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value '{value}'.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be greater than 0.")
    return parsed


def validate_date_range(start_date: dt.date, end_date: dt.date) -> None:
    if end_date < start_date:
        raise argparse.ArgumentTypeError(
            f"Invalid date range: end_date={end_date} is earlier than start_date={start_date}."
        )


def shift_date_years(value: dt.date, years: int) -> dt.date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        # Handles leap day when the target year is not a leap year.
        return value.replace(year=value.year + years, day=28)


def resolve_date_range(args: argparse.Namespace, defaults: Config) -> tuple[dt.date, dt.date]:
    if args.backfill_years is not None:
        if args.backfill_years < 1:
            raise argparse.ArgumentTypeError("--backfill-years must be greater than 0.")
        if args.start_date is not None or args.end_date is not None:
            raise argparse.ArgumentTypeError(
                "--backfill-years cannot be combined with --start-date or --end-date."
            )

        start_date = shift_date_years(defaults.START_DATE, -args.backfill_years)
        end_date = defaults.START_DATE - dt.timedelta(days=1)
        validate_date_range(start_date, end_date)
        return start_date, end_date

    start_date = args.start_date or defaults.START_DATE
    end_date = args.end_date or defaults.END_DATE
    validate_date_range(start_date, end_date)
    return start_date, end_date


def resolve_output_tag(
    args: argparse.Namespace,
    ticker: str,
    start_date: dt.date,
    end_date: dt.date,
) -> str | None:
    if args.output_tag:
        return args.output_tag
    if args.backfill_years is None:
        return None
    return f"{ticker.lower()}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"


def resolve_google_trends_reference_range(
    args: argparse.Namespace,
    start_date: dt.date,
    end_date: dt.date,
    defaults: Config,
) -> tuple[dt.date | None, dt.date | None]:
    reference_start = args.google_trends_reference_start_date
    reference_end = args.google_trends_reference_end_date

    if args.backfill_years is not None:
        reference_start = reference_start or start_date
        reference_end = reference_end or defaults.END_DATE

    if (reference_start is None) != (reference_end is None):
        raise argparse.ArgumentTypeError(
            "Both --google-trends-reference-start-date and "
            "--google-trends-reference-end-date must be provided together."
        )
    if reference_start is None or reference_end is None:
        return None, None

    validate_date_range(reference_start, reference_end)
    if reference_start > start_date or reference_end < end_date:
        raise argparse.ArgumentTypeError(
            "Google Trends reference range must cover the requested import range."
        )
    return reference_start, reference_end


def build_config(args: argparse.Namespace) -> Config:
    defaults = Config()
    ticker = (args.ticker or defaults.TICKER).strip().upper()
    start_date, end_date = resolve_date_range(args, defaults)
    reference_start, reference_end = resolve_google_trends_reference_range(
        args,
        start_date,
        end_date,
        defaults,
    )

    return build_profiled_config(
        ticker=ticker,
        company_name=args.company_name,
        trends_query=args.trends_query,
        gdelt_query=args.gdelt_query,
        geo=args.geo or defaults.GEO,
        google_trends_window_days=args.google_trends_window_days,
        google_trends_reference_scaling=args.google_trends_reference_scaling,
        google_trends_reference_start_date=reference_start,
        google_trends_reference_end_date=reference_end,
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
        output_tag=resolve_output_tag(args, ticker, start_date, end_date),
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
    no_overwrite: bool = False,
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

        existing_outputs = tuple(path for path in expected_outputs if path.exists())
        if no_overwrite and existing_outputs:
            message = (
                f"Refusing to run importer: {importer.name}. "
                f"Existing output file(s) would be overwritten: "
                f"{', '.join(str(path) for path in existing_outputs)}"
            )
            if not continue_on_error:
                raise FileExistsError(message)
            print(message)
            results[importer.name] = {"error": message}
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
    start_date, end_date = resolve_date_range(args, defaults)
    reference_start, reference_end = resolve_google_trends_reference_range(
        args,
        start_date,
        end_date,
        defaults,
    )

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
            google_trends_window_days=args.google_trends_window_days,
            google_trends_reference_scaling=args.google_trends_reference_scaling,
            google_trends_reference_start_date=reference_start,
            google_trends_reference_end_date=reference_end,
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
            output_tag=resolve_output_tag(args, ticker, start_date, end_date),
        )
        for ticker in tickers
    ]


def run_batch(
    configs: list[Config],
    selected: list[str] | None = None,
    skip_existing: bool = False,
    continue_on_error: bool = False,
    no_overwrite: bool = False,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for index, config in enumerate(configs, start=1):
        print(f"\n=== [{index}/{len(configs)}] {config.TICKER} / {config.COMPANY_NAME} ===")
        results[config.TICKER] = run_importers(
            selected=selected,
            config=config,
            skip_existing=skip_existing,
            continue_on_error=continue_on_error,
            no_overwrite=no_overwrite,
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
        "--google-trends-window-days",
        type=positive_int,
        help="Daily Google Trends request window size. Default: 200.",
    )
    parser.add_argument(
        "--google-trends-reference-scaling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Scale daily Google Trends windows to a full-period reference series "
            "(default: enabled). Use --no-google-trends-reference-scaling to keep raw chunks."
        ),
    )
    parser.add_argument(
        "--google-trends-reference-start-date",
        type=parse_date,
        help=(
            "Start date for the common Google Trends reference scale. "
            "Use this when importing split periods that must share one scale."
        ),
    )
    parser.add_argument(
        "--google-trends-reference-end-date",
        type=parse_date,
        help=(
            "End date for the common Google Trends reference scale. "
            "Use this when importing split periods that must share one scale."
        ),
    )
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
        "--backfill-years",
        type=int,
        help=(
            "Fetch N full years before the default START_DATE into date-tagged output files. "
            "For the current default START_DATE=2023-01-01, --backfill-years 2 uses "
            "2021-01-01 through 2022-12-31."
        ),
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
        "--no-overwrite",
        action="store_true",
        help="Abort before running an importer if any expected output file already exists.",
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
    no_overwrite = args.no_overwrite or args.backfill_years is not None

    if len(configs) == 1:
        return run_importers(
            args.importers,
            config=configs[0],
            skip_existing=args.skip_existing,
            continue_on_error=args.continue_on_error,
            no_overwrite=no_overwrite,
        )
    return run_batch(
        configs,
        selected=args.importers,
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
        no_overwrite=no_overwrite,
    )


if __name__ == "__main__":
    main()
