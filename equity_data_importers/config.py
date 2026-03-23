import datetime as dt
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    COMPANY_NAME: str = "Tesla"
    TICKER: str = "TSLA"
    GDELT_QUERY: str = "(Tesla OR TSLA)"
    GEO: str = "US"
    START_DATE: dt.date = dt.date(2023, 1, 1)
    END_DATE: dt.date = dt.date(2025, 12, 31)
