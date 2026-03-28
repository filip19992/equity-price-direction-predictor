import datetime as dt
import json
import random
import re
import time
import urllib.parse
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
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

    def __init__(self, config: type[Config] = Config) -> None:
        super().__init__(config=config)
        self.gdelt_request_spacing = 15
        self._last_gdelt_request_at = 0.0
        self.cache_dir = self.data_dir / "gdelt_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_metric(
        self,
        query: str,
        start_date: dt.date,
        end_date: dt.date,
        geo: str | None,
        mode: str,
        value_name: str,
        retry_delay: int = 5,
        max_attempts: int = 6,
        window_days: int = 30,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        current_start = start_date
        window = dt.timedelta(days=window_days)

        while current_start <= end_date:
            current_end = min(current_start + window, end_date)
            frame = self.fetch_metric_window(
                query=query,
                start_date=current_start,
                end_date=current_end,
                geo=geo,
                mode=mode,
                value_name=value_name,
                retry_delay=retry_delay,
                max_attempts=max_attempts,
            )
            frames.append(frame)

            current_start = current_end + dt.timedelta(days=1)
            if current_start <= end_date:
                time.sleep(retry_delay)

        merged = pd.concat(frames).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        return merged

    def get_cache_path(self, mode: str, start_date: dt.date, end_date: dt.date) -> Path:
        return self.cache_dir / (
            f"{mode}_{start_date.isoformat()}_{end_date.isoformat()}.csv"
        )

    def wait_for_request_slot(self) -> None:
        elapsed = time.time() - self._last_gdelt_request_at
        if elapsed < self.gdelt_request_spacing:
            time.sleep(self.gdelt_request_spacing - elapsed)

    def get_retry_wait_time(
        self,
        retry_delay: int,
        attempt: int,
        response: requests.Response | None,
    ) -> int:
        retry_after = None
        if response is not None:
            retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return max(int(retry_after), self.gdelt_request_spacing)
        return max(retry_delay * (2**attempt), self.gdelt_request_spacing)

    def fetch_metric_window(
        self,
        query: str,
        start_date: dt.date,
        end_date: dt.date,
        geo: str | None,
        mode: str,
        value_name: str,
        retry_delay: int,
        max_attempts: int,
    ) -> pd.DataFrame:
        cache_path = self.get_cache_path(mode, start_date, end_date)
        if cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")

        try:
            frame = self.fetch_metric_window_once(
                query=query,
                start_date=start_date,
                end_date=end_date,
                geo=geo,
                mode=mode,
                value_name=value_name,
                retry_delay=retry_delay,
                max_attempts=max_attempts,
            )
        except RuntimeError:
            if (end_date - start_date).days <= 1:
                raise

            midpoint = start_date + (end_date - start_date) / 2
            left_end = midpoint
            right_start = midpoint + dt.timedelta(days=1)
            print(
                f"Splitting GDELT window for {mode}: "
                f"{start_date} to {end_date} into "
                f"{start_date} to {left_end} and {right_start} to {end_date}"
            )
            left = self.fetch_metric_window(
                query=query,
                start_date=start_date,
                end_date=left_end,
                geo=geo,
                mode=mode,
                value_name=value_name,
                retry_delay=retry_delay,
                max_attempts=max_attempts,
            )
            right = self.fetch_metric_window(
                query=query,
                start_date=right_start,
                end_date=end_date,
                geo=geo,
                mode=mode,
                value_name=value_name,
                retry_delay=retry_delay,
                max_attempts=max_attempts,
            )
            frame = pd.concat([left, right]).sort_index()
            frame = frame[~frame.index.duplicated(keep="last")]

        frame.to_csv(cache_path, index=True)
        return frame

    def fetch_metric_window_once(
        self,
        query: str,
        start_date: dt.date,
        end_date: dt.date,
        geo: str | None,
        mode: str,
        value_name: str,
        retry_delay: int,
        max_attempts: int,
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
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                self.wait_for_request_slot()
                response = requests.get(url, timeout=30)
                self._last_gdelt_request_at = time.time()
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                if response is not None and response.status_code == 429:
                    wait_time = self.get_retry_wait_time(
                        retry_delay=retry_delay,
                        attempt=attempt,
                        response=response,
                    )
                    print(
                        f"GDELT rate limited for {mode} "
                        f"{start_date} to {end_date}; waiting {wait_time}s"
                    )
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.RequestException as exc:
                last_error = exc
                wait_time = max(retry_delay * (2**attempt), self.gdelt_request_spacing)
                print(
                    f"GDELT request failed for {mode} "
                    f"{start_date} to {end_date}; waiting {wait_time}s"
                )
                time.sleep(wait_time)

        if response is None or response.status_code >= 400:
            raise RuntimeError(
                "Failed to fetch GDELT data for "
                f"mode={mode}, start_date={start_date}, end_date={end_date}"
            ) from last_error

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
    market_timezone = ZoneInfo("America/New_York")
    market_open_time = dt.time(9, 30)
    finbert_model_name = "ProsusAI/finbert"
    finbert_batch_size = 32
    line_progress_interval = 250000

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
        self._finbert_components: tuple[object, object, object] | None = None

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

    @staticmethod
    def sanitize_count(value: object) -> float:
        if pd.isna(value):
            return 0.0
        return max(float(value), 0.0)

    def get_trading_sessions(self) -> list[dt.date]:
        history_start = self.config.START_DATE - dt.timedelta(days=10)
        history_end = self.config.END_DATE + dt.timedelta(days=10)
        sessions = yf.download(
            self.config.TICKER,
            start=str(history_start),
            end=str(history_end),
            progress=False,
        )
        if sessions.empty:
            raise RuntimeError(
                f"Unable to load trading sessions for {self.config.TICKER}"
            )

        return [ts.date() for ts in pd.to_datetime(sessions.index).to_pydatetime()]

    def align_to_next_trading_session(
        self,
        created_utc: int,
        trading_sessions: list[dt.date],
    ) -> str | None:
        timestamp = pd.Timestamp(created_utc, unit="s", tz="UTC").tz_convert(
            self.market_timezone
        )
        local_date = timestamp.date()

        for session_date in trading_sessions:
            if session_date < local_date:
                continue
            if session_date == local_date and timestamp.time() < self.market_open_time:
                return session_date.isoformat()
            if session_date > local_date:
                return session_date.isoformat()

        return None

    def extract_matching_posts(self) -> pd.DataFrame:
        rows = []
        processed = 0
        matched = 0
        source_start_date = self.config.START_DATE - dt.timedelta(days=7)
        print(f"Scanning reddit source file: {self.source_path}")

        for obj in self.read_ndjson_plain(self.source_path):
            processed += 1
            created_utc = obj.get("created_utc")
            if created_utc is None:
                continue

            if processed % self.line_progress_interval == 0:
                print(
                    f"Scanned {processed:,} lines, matched {matched:,} posts so far"
                )

            current_date = self.to_utc_date(created_utc)
            if current_date < source_start_date or current_date > self.config.END_DATE:
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
        print(
            f"Finished reddit scan: processed {processed:,} lines, matched {matched:,} posts"
        )
        return pd.DataFrame(rows, columns=columns)

    def load_finbert(self) -> tuple[object, object, object]:
        if self._finbert_components is not None:
            return self._finbert_components

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "FinBERT scoring requires a working 'torch' and 'transformers' setup."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.finbert_model_name
        )
        model.eval()
        self._finbert_components = (torch, tokenizer, model)
        return self._finbert_components

    def score_finbert(self, texts: pd.Series) -> pd.Series:
        torch, tokenizer, model = self.load_finbert()
        values: list[float] = []
        total_batches = (len(texts) + self.finbert_batch_size - 1) // self.finbert_batch_size
        print(
            f"Scoring FinBERT sentiment for {len(texts):,} posts "
            f"in {total_batches:,} batches"
        )

        for batch_index, start in enumerate(
            range(0, len(texts), self.finbert_batch_size),
            start=1,
        ):
            batch = texts.iloc[start : start + self.finbert_batch_size].tolist()
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            labels = [model.config.id2label[i].lower() for i in range(probabilities.shape[1])]
            label_to_index = {label: idx for idx, label in enumerate(labels)}
            positive_idx = label_to_index.get("positive")
            negative_idx = label_to_index.get("negative")

            if positive_idx is None or negative_idx is None:
                raise RuntimeError(
                    f"Unexpected FinBERT labels: {model.config.id2label}"
                )

            values.extend(
                (
                    probabilities[:, positive_idx] - probabilities[:, negative_idx]
                ).tolist()
            )

            print(f"FinBERT progress: batch {batch_index:,}/{total_batches:,}")

        return pd.Series(values, index=texts.index, dtype="float64")

    def enrich_posts(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        trading_sessions = self.get_trading_sessions()
        working = frame.copy()
        print(f"Aligning {len(working):,} matched posts to trading sessions")
        working["aligned_date"] = working["created_utc"].map(
            lambda created_utc: self.align_to_next_trading_session(
                created_utc=created_utc,
                trading_sessions=trading_sessions,
            )
        )
        working = working[working["aligned_date"].notna()].copy()
        working["aligned_date"] = working["aligned_date"].astype(str)
        working["text"] = working["title"].fillna("") + "\n" + working["selftext"].fillna("")
        print(f"Posts remaining after session alignment: {len(working):,}")

        analyzer = SentimentIntensityAnalyzer()
        print("Scoring VADER sentiment")
        working["vader_sentiment"] = working["text"].map(
            lambda text: analyzer.polarity_scores(text)["compound"]
        )
        try:
            working["finbert_sentiment"] = self.score_finbert(working["text"])
        except Exception as exc:
            if self.config.FINBERT_REQUIRED:
                raise
            print(f"FinBERT unavailable, continuing with VADER only: {exc}")
            working["finbert_sentiment"] = np.nan

        score_weight = working["score"].map(self.sanitize_count).map(np.log1p)
        comments_weight = working["num_comments"].map(self.sanitize_count).map(np.log1p)
        # Keep an equal base weight so zero-engagement posts still contribute tone.
        working["engagement_weight"] = 1.0 + score_weight + comments_weight
        working["vader_weighted_sentiment"] = (
            working["vader_sentiment"] * working["engagement_weight"]
        )
        working["finbert_weighted_sentiment"] = (
            working["finbert_sentiment"] * working["engagement_weight"]
        )

        return working

    def aggregate_daily_metrics(self, frame: pd.DataFrame) -> pd.DataFrame:
        print("Aggregating daily reddit metrics")
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
            calendar["reddit_sent_std"] = pd.NA
            calendar["reddit_weight_sum"] = 0.0
            calendar["reddit_score_sum"] = 0.0
            calendar["reddit_comments_sum"] = 0.0
            calendar["reddit_vader_mean"] = pd.NA
            calendar["reddit_vader_weighted_mean"] = pd.NA
            calendar["reddit_vader_sum"] = 0.0
            calendar["reddit_vader_std"] = pd.NA
            calendar["reddit_finbert_mean"] = pd.NA
            calendar["reddit_finbert_weighted_mean"] = pd.NA
            calendar["reddit_finbert_sum"] = 0.0
            calendar["reddit_finbert_std"] = pd.NA
            return calendar

        working = frame.copy()
        working["date"] = working["aligned_date"]

        daily = working.groupby("date", as_index=False).agg(
            reddit_posts=("id", "count"),
            reddit_weight_sum=("engagement_weight", "sum"),
            reddit_score_sum=("score", "sum"),
            reddit_comments_sum=("num_comments", "sum"),
            reddit_vader_mean=("vader_sentiment", "mean"),
            reddit_vader_sum=("vader_sentiment", "sum"),
            reddit_vader_std=("vader_sentiment", "std"),
            reddit_vader_weighted_sum=("vader_weighted_sentiment", "sum"),
            reddit_finbert_mean=("finbert_sentiment", "mean"),
            reddit_finbert_sum=("finbert_sentiment", "sum"),
            reddit_finbert_std=("finbert_sentiment", "std"),
            reddit_finbert_weighted_sum=("finbert_weighted_sentiment", "sum"),
        )
        daily["reddit_vader_weighted_mean"] = (
            daily["reddit_vader_weighted_sum"] / daily["reddit_weight_sum"]
        )
        daily["reddit_finbert_weighted_mean"] = (
            daily["reddit_finbert_weighted_sum"] / daily["reddit_weight_sum"]
        )
        daily["reddit_sent_mean"] = daily["reddit_vader_mean"]
        daily["reddit_sent_sum"] = daily["reddit_vader_sum"]
        daily["reddit_sent_std"] = daily["reddit_vader_std"]

        daily_full = calendar.merge(daily, on="date", how="left")
        daily_full["reddit_posts"] = daily_full["reddit_posts"].fillna(0).astype(int)
        daily_full["reddit_weight_sum"] = daily_full["reddit_weight_sum"].fillna(0)
        daily_full["reddit_score_sum"] = daily_full["reddit_score_sum"].fillna(0)
        daily_full["reddit_comments_sum"] = daily_full["reddit_comments_sum"].fillna(0)
        daily_full["reddit_sent_sum"] = daily_full["reddit_sent_sum"].fillna(0)
        daily_full["reddit_vader_sum"] = daily_full["reddit_vader_sum"].fillna(0)
        daily_full["reddit_finbert_sum"] = daily_full["reddit_finbert_sum"].fillna(0)
        daily_full = daily_full.drop(
            columns=["reddit_vader_weighted_sum", "reddit_finbert_weighted_sum"],
            errors="ignore",
        )
        return daily_full

    def run(self) -> tuple[Path, Path]:
        posts = self.extract_matching_posts()
        posts = self.enrich_posts(posts)
        print(f"Writing {len(posts):,} matched reddit posts to {self.raw_output_path}")
        posts.to_parquet(self.raw_output_path, index=False)
        print(f"Saved reddit post matches to {self.raw_output_path}")

        daily = self.aggregate_daily_metrics(posts)
        print(f"Writing {len(daily):,} daily reddit rows to {self.daily_output_path}")
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
