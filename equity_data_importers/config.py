import datetime as dt
import re
from dataclasses import dataclass


BIG_TECH_10_TICKERS = (
    "NVDA",
    "MSFT",
    "AAPL",
    "AMD",
    "TSLA",
    "NFLX",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
)

TICKER_GROUPS: dict[str, tuple[str, ...]] = {
    "big_tech_10": BIG_TECH_10_TICKERS,
}

TICKER_PROFILES: dict[str, dict[str, str]] = {
    "AAPL": {
        "COMPANY_NAME": "Apple",
        "TRENDS_QUERY": "Apple stock",
        "GDELT_QUERY": '("Apple Inc" OR AAPL)',
    },
    "AMZN": {
        "COMPANY_NAME": "Amazon",
        "TRENDS_QUERY": "Amazon stock",
        "GDELT_QUERY": '("Amazon.com" OR AMZN)',
    },
    "AMD": {
        "COMPANY_NAME": "AMD",
        "TRENDS_QUERY": "AMD stock",
        "GDELT_QUERY": '"Advanced Micro Devices"',
    },
    "AVGO": {
        "COMPANY_NAME": "Broadcom",
        "TRENDS_QUERY": "Broadcom stock",
        "GDELT_QUERY": '("Broadcom" OR AVGO)',
    },
    "GOOGL": {
        "COMPANY_NAME": "Google",
        "TRENDS_QUERY": "Google stock",
        "GDELT_QUERY": '("Google" OR "Alphabet" OR GOOGL)',
    },
    "META": {
        "COMPANY_NAME": "Meta",
        "TRENDS_QUERY": "Meta stock",
        "GDELT_QUERY": '("Meta Platforms" OR META OR Facebook)',
    },
    "MSFT": {
        "COMPANY_NAME": "Microsoft",
        "TRENDS_QUERY": "Microsoft stock",
        "GDELT_QUERY": '("Microsoft" OR MSFT)',
    },
    "NFLX": {
        "COMPANY_NAME": "Netflix",
        "TRENDS_QUERY": "Netflix stock",
        "GDELT_QUERY": '("Netflix" OR NFLX)',
    },
    "NVDA": {
        "COMPANY_NAME": "NVIDIA",
        "TRENDS_QUERY": "NVIDIA stock",
        "GDELT_QUERY": '("NVIDIA" OR NVDA)',
    },
    "TSLA": {
        "COMPANY_NAME": "Tesla",
        "TRENDS_QUERY": "Tesla stock",
        "GDELT_QUERY": '("Tesla" OR TSLA)',
    },
}

@dataclass(frozen=True)
class Config:
    COMPANY_NAME: str = "Tesla"
    TICKER: str = "TSLA"
    GDELT_QUERY: str | None = None
    TRENDS_QUERY: str | None = None
    GEO: str = "US"
    START_DATE: dt.date = dt.date(2023, 1, 1)
    END_DATE: dt.date = dt.date(2025, 12, 31)
    FINBERT_REQUIRED: bool = True
    REDDIT_SUBMISSIONS_SOURCE: str = "stocks_submissions"
    REDDIT_COMMENTS_SOURCE: str = "stocks_comments"
    OUTPUT_TAG: str | None = None

    @property
    def resolved_trends_query(self) -> str:
        value = (self.TRENDS_QUERY or "").strip()
        return value if value else self.COMPANY_NAME

    @property
    def resolved_gdelt_query(self) -> str:
        value = (self.GDELT_QUERY or "").strip()
        if value:
            return value
        if self.COMPANY_NAME.strip().upper() == self.TICKER.upper():
            return f"({self.TICKER})"
        return f"({self.COMPANY_NAME} OR {self.TICKER})"

    @property
    def resolved_output_tag(self) -> str:
        raw = (self.OUTPUT_TAG or self.TICKER).strip().lower()
        cleaned = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-")
        return cleaned if cleaned else self.TICKER.lower()

    def is_legacy_default_profile(self) -> bool:
        return (
            self.TICKER.upper() == "TSLA"
            and self.COMPANY_NAME.strip().lower() == "tesla"
            and (self.OUTPUT_TAG is None or self.OUTPUT_TAG.strip() == "")
        )


def parse_ticker_values(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if not values:
        return []

    seen: set[str] = set()
    tickers: list[str] = []
    for value in values:
        for raw in re.split(r"[\s,]+", value.strip()):
            ticker = raw.strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def get_group_tickers(group_name: str) -> list[str]:
    try:
        return list(TICKER_GROUPS[group_name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown ticker group '{group_name}'. Available groups: {', '.join(sorted(TICKER_GROUPS))}."
        ) from exc


def build_profiled_config(
    ticker: str,
    *,
    company_name: str | None = None,
    trends_query: str | None = None,
    gdelt_query: str | None = None,
    geo: str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    finbert_required: bool | None = None,
    reddit_submissions_source: str | None = None,
    reddit_comments_source: str | None = None,
    output_tag: str | None = None,
) -> Config:
    defaults = Config()
    resolved_ticker = ticker.strip().upper()
    profile = TICKER_PROFILES.get(resolved_ticker, {})

    if company_name is not None:
        resolved_company_name = company_name.strip()
    else:
        resolved_company_name = profile.get(
            "COMPANY_NAME",
            defaults.COMPANY_NAME if resolved_ticker == defaults.TICKER else resolved_ticker,
        )

    return Config(
        COMPANY_NAME=resolved_company_name,
        TICKER=resolved_ticker,
        GDELT_QUERY=gdelt_query if gdelt_query is not None else profile.get("GDELT_QUERY"),
        TRENDS_QUERY=trends_query if trends_query is not None else profile.get("TRENDS_QUERY"),
        GEO=(geo if geo is not None else defaults.GEO).strip(),
        START_DATE=start_date or defaults.START_DATE,
        END_DATE=end_date or defaults.END_DATE,
        FINBERT_REQUIRED=(
            defaults.FINBERT_REQUIRED if finbert_required is None else finbert_required
        ),
        REDDIT_SUBMISSIONS_SOURCE=reddit_submissions_source or defaults.REDDIT_SUBMISSIONS_SOURCE,
        REDDIT_COMMENTS_SOURCE=reddit_comments_source or defaults.REDDIT_COMMENTS_SOURCE,
        OUTPUT_TAG=output_tag,
    )
