from equity_data_importers.importers import (
    GdeltImporter,
    GoogleTrendsImporter,
    RedditImporter,
    StockPriceImporter,
)


def run_all() -> dict[str, object]:
    importers = [
        GoogleTrendsImporter(),
        GdeltImporter(),
        RedditImporter(),
        StockPriceImporter(),
    ]
    results: dict[str, object] = {}

    for importer in importers:
        print(f"Running importer: {importer.name}")
        results[importer.name] = importer.run()

    return results


if __name__ == "__main__":
    run_all()
