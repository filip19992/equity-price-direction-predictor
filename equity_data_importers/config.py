import datetime as dt
import re
from dataclasses import dataclass

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
