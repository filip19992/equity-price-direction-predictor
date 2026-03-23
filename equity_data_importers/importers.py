import datetime as dt
import json
import random
import re
import time
import urllib.parse
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from pytrends.request import TrendReq
from sklearn.preprocessing import RobustScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from equity_data_importers.config import Config


class BaseImporter(ABC):
    name = "base"

    def __init__(self, config: type[Config] = Config) -> None:
        self.config = config
        self.data_dir = Path(__file__).resolve().parent.parent / "data" / "equity_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self) -> Path | tuple[Path, ...]:
        raise NotImplementedError


class GoogleTrendsImporter(BaseImporter):
    name = "google_trends"

    def fetch_google_trends(
        self,
        query: str,
        start_date: dt.date,
        end_date: dt.date,
        geo: str | None = None,
        window_days: int = 200,
    ) -> pd.DataFrame:
        delta = dt.timedelta(days=window_days)
        current_start = start_date
        collected: list[pd.DataFrame] = []
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (X11; Linux x86_64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        ]

        print(f"Fetching Google Trends for: {query}")
        while current_start <= end_date:
            current_end = min(current_start + delta, end_date)
            timeframe = f"{current_start:%Y-%m-%d} {current_end:%Y-%m-%d}"

            try:
                headers = {"User-Agent": random.choice(user_agents)}
                trends = TrendReq(
                    hl="en-US",
                    tz=360,
                    retries=3,
                    backoff_factor=0.5,
                    requests_args={"headers": headers},
                )
                trends.build_payload([query], timeframe=timeframe, geo=geo)
                frame = trends.interest_over_time().drop(
                    columns="isPartial", errors="ignore"
                )

                if frame.empty:
                    print(f"No Google Trends data for {timeframe}")
                else:
                    collected.append(frame)
                    print(f"Google Trends {timeframe}: {len(frame)} rows")
            except Exception as exc:
                print(f"Google Trends error for {timeframe}: {exc}")

            current_start = current_end + dt.timedelta(days=1)
            if current_start <= end_date:
                wait_seconds = random.randint(60, 90)
                print(f"Waiting {wait_seconds}s before the next Trends request")
                time.sleep(wait_seconds)

        if not collected:
            return pd.DataFrame(
                columns=["trends_score"],
                index=pd.date_range(start_date, end_date, freq="D"),
            )

        full = pd.concat(collected).sort_index()
        full = full[~full.index.duplicated()]
        full = full.rename(columns={query: "trends_score"})
        return full[["trends_score"]]

    def run(self) -> Path:
        frame = self.fetch_google_trends(
            query=self.config.COMPANY_NAME,
            start_date=self.config.START_DATE,
            end_date=self.config.END_DATE,
            geo=self.config.GEO,
            window_days=200,
        )
        output_path = self.data_dir / "google_trends_data.csv"
        frame.to_csv(output_path, index=True)
        print(f"Saved Google Trends data to {output_path}")
        return output_path


class GdeltImporter(BaseImporter):
    name = "gdelt"

    def fetch_metric(
        self,
        query: str,
        start_date: dt.date,
        end_date: dt.date,
        geo: str | None,
        mode: str,
        value_name: str,
        retry_delay: int = 5,
    ) -> pd.DataFrame:
        params = {
            "query": query,
            "mode": mode,
            "format": "csv",
            "startdatetime": start_date.strftime("%Y%m%d") + "000000",
            "enddatetime": end_date.strftime("%Y%m%d") + "000000",
        }
        if geo:
            params["geo"] = geo

        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc?"
            + urllib.parse.urlencode(params)
        )
        response = None

        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError:
                if response is not None and response.status_code == 429:
                    wait_time = retry_delay * (2**attempt)
                    print(f"GDELT rate limited for {mode}; waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                raise

        if response is None or response.status_code >= 400:
            raise RuntimeError(f"Failed to fetch GDELT data for mode={mode}")

        lines = response.text.splitlines()
        if lines and lines[0].startswith("sep="):
            lines = lines[1:]
        cleaned = "\n".join(lines)

        frame = pd.read_csv(
            StringIO(cleaned), header=0, parse_dates=["Date"], index_col="Date"
        )
        if "Series" in frame.columns:
            frame = frame.drop(columns=["Series"])
        frame = frame.rename(columns={"Value": value_name})
        time.sleep(retry_delay)
        return frame

    def run(self) -> Path:
        volume = self.fetch_metric(
            query=self.config.GDELT_QUERY,
            start_date=self.config.START_DATE,
            end_date=self.config.END_DATE,
            geo=self.config.GEO,
            mode="TimelineVol",
            value_name="gdelt_articles",
        )
        tone = self.fetch_metric(
            query=self.config.GDELT_QUERY,
            start_date=self.config.START_DATE,
            end_date=self.config.END_DATE,
            geo=self.config.GEO,
            mode="TimelineTone",
            value_name="sentiment_score",
        )

        scaler = RobustScaler()
        volume["gdelt_robust"] = scaler.fit_transform(volume[["gdelt_articles"]])
        merged = volume.join(tone, how="left")

        output_path = self.data_dir / "gdelt_data.csv"
        merged.to_csv(output_path, index=True)
        print(f"Saved GDELT data to {output_path}")
        return output_path


class RedditImporter(BaseImporter):
    name = "reddit"

    def __init__(self, config: type[Config] = Config) -> None:
        super().__init__(config=config)
        pattern = (
            rf"(?i)\b({re.escape(self.config.TICKER)}|\${re.escape(self.config.TICKER)}|"
            rf"{re.escape(self.config.COMPANY_NAME)})\b"
        )
        self.keyword_pattern = re.compile(pattern)
        self.source_path = self.data_dir / "stocks_submissions"
        self.raw_output_path = self.data_dir / "tesla_stocks_posts.parquet"
        self.daily_output_path = self.data_dir / "stock-reddit-data.csv"

    def read_ndjson_plain(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Reddit source file not found: {path}")
        with path.open("rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def to_utc_date(created_utc: int) -> dt.date:
        return dt.datetime.utcfromtimestamp(int(created_utc)).date()

    def extract_matching_posts(self) -> pd.DataFrame:
        rows = []
        processed = 0
        matched = 0

        for obj in self.read_ndjson_plain(self.source_path):
            processed += 1
            created_utc = obj.get("created_utc")
            if created_utc is None:
                continue

            current_date = self.to_utc_date(created_utc)
            if current_date < self.config.START_DATE or current_date >= self.config.END_DATE:
                continue

            title = obj.get("title") or ""
            selftext = obj.get("selftext") or ""
            text = f"{title}\n{selftext}"

            if not self.keyword_pattern.search(text):
                continue

            matched += 1
            rows.append(
                {
                    "id": obj.get("id"),
                    "date_utc": current_date.isoformat(),
                    "created_utc": int(created_utc),
                    "subreddit": obj.get("subreddit"),
                    "title": title,
                    "selftext": selftext,
                    "score": obj.get("score"),
                    "num_comments": obj.get("num_comments"),
                    "permalink": obj.get("permalink"),
                    "url": obj.get("url"),
                }
            )

            if matched % 50000 == 0:
                print(f"Matched {matched:,} reddit posts after {processed:,} lines")

        columns = [
            "id",
            "date_utc",
            "created_utc",
            "subreddit",
            "title",
            "selftext",
            "score",
            "num_comments",
            "permalink",
            "url",
        ]
        return pd.DataFrame(rows, columns=columns)

    def aggregate_daily_metrics(self, frame: pd.DataFrame) -> pd.DataFrame:
        calendar = pd.DataFrame(
            {
                "date": pd.date_range(
                    self.config.START_DATE,
                    self.config.END_DATE - dt.timedelta(days=1),
                    freq="D",
                ).strftime("%Y-%m-%d")
            }
        )

        if frame.empty:
            calendar["reddit_posts"] = 0
            calendar["reddit_sent_mean"] = pd.NA
            calendar["reddit_sent_sum"] = 0.0
            calendar["reddit_score_sum"] = 0.0
            calendar["reddit_comments_sum"] = 0.0
            return calendar

        working = frame.copy()
        working["date"] = pd.to_datetime(working["date_utc"]).dt.strftime("%Y-%m-%d")
        working["text"] = working["title"].fillna("") + "\n" + working["selftext"].fillna("")

        analyzer = SentimentIntensityAnalyzer()
        working["sentiment"] = working["text"].map(
            lambda text: analyzer.polarity_scores(text)["compound"]
        )

        daily = working.groupby("date", as_index=False).agg(
            reddit_posts=("id", "count"),
            reddit_sent_mean=("sentiment", "mean"),
            reddit_sent_sum=("sentiment", "sum"),
            reddit_score_sum=("score", "sum"),
            reddit_comments_sum=("num_comments", "sum"),
        )

        daily_full = calendar.merge(daily, on="date", how="left")
        daily_full["reddit_posts"] = daily_full["reddit_posts"].fillna(0).astype(int)
        daily_full["reddit_score_sum"] = daily_full["reddit_score_sum"].fillna(0)
        daily_full["reddit_comments_sum"] = daily_full["reddit_comments_sum"].fillna(0)
        daily_full["reddit_sent_sum"] = daily_full["reddit_sent_sum"].fillna(0)
        return daily_full

    def run(self) -> tuple[Path, Path]:
        posts = self.extract_matching_posts()
        posts.to_parquet(self.raw_output_path, index=False)
        print(f"Saved reddit post matches to {self.raw_output_path}")

        daily = self.aggregate_daily_metrics(posts)
        daily.to_csv(self.daily_output_path, index=False)
        print(f"Saved reddit daily aggregates to {self.daily_output_path}")
        return self.raw_output_path, self.daily_output_path


class StockPriceImporter(BaseImporter):
    name = "stock_price"

    def run(self) -> Path:
        yf_end = self.config.END_DATE + dt.timedelta(days=1)
        print(f"Fetching stock price data for {self.config.TICKER}")
        raw = yf.download(
            self.config.TICKER,
            start=str(self.config.START_DATE),
            end=str(yf_end),
            progress=False,
        )

        if raw.empty or "Close" not in raw.columns or "Volume" not in raw.columns:
            raise RuntimeError("Stock price data is empty or missing Close/Volume columns")

        stock_frame = raw[["Close", "Volume"]].copy()
        stock_frame.columns = ["stock_price", "stock_volume"]

        output_path = self.data_dir / "stock-prices-data.csv"
        stock_frame.to_csv(output_path, index=True)
        print(f"Saved stock price data to {output_path}")
        return output_path
