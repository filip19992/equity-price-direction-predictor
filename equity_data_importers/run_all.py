import argparse

from equity_data_importers.importers import (
    GdeltImporter,
    GoogleTrendsImporter,
    RedditImporter,
    StockPriceImporter,
)

IMPORTERS = {
    "google_trends": GoogleTrendsImporter,
    "gdelt": GdeltImporter,
    "reddit": RedditImporter,
    "stock_price": StockPriceImporter,
}


def run_importers(selected: list[str] | None = None) -> dict[str, object]:
    importer_names = selected or list(IMPORTERS.keys())
    importers = [IMPORTERS[name]() for name in importer_names]
    results: dict[str, object] = {}

    for importer in importers:
        print(f"Running importer: {importer.name}")
        results[importer.name] = importer.run()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "importers",
        nargs="*",
        choices=sorted(IMPORTERS.keys()),
        help="Optional importer names to run. If omitted, all importers run.",
    )
    return parser.parse_args()


def run_all() -> dict[str, object]:
    return run_importers()


def main() -> dict[str, object]:
    args = parse_args()
    return run_importers(args.importers)


if __name__ == "__main__":
    main()
