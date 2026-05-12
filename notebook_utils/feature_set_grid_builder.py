from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSetGrid:
    price_features: list[str]
    volume_feature_options: dict[str, list[str]]
    gdelt_feature_options: dict[str, list[str]]
    gdelt_sentiment_feature_options: dict[str, list[str]]
    gdelt_attention_feature_options: dict[str, list[str]]
    reddit_feature_options: dict[str, list[str]]
    reddit_attention_feature_options: dict[str, list[str]]
    google_trends_feature_options: dict[str, list[str]]
    google_score_attention_feature_options: dict[str, list[str]]
    derived_feature_columns: set[str]
    base_volume_option: str
    baseline_feature_set: str
    feature_set_specs: list[dict]
    feature_sets: dict[str, list[str]]
    feature_set_metadata: dict[str, dict]
    skipped_feature_sets: list[dict]
    feature_sets_to_test: list[str]

    def candidate_feature_sets_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.feature_set_metadata.values()).sort_values(
            ["feature_family", "feature_set"]
        ).reset_index(drop=True)


class FeatureSetGridBuilder:
    """Build the compact feature-set grids shared by the model notebooks."""

    PRICE_FEATURES = [
        "return_1d",
        "return_5d",
        "return_20d",
        "rolling_volatility_20d",
    ]

    BASE_VOLUME_OPTION = "volume zscore 20d"

    @classmethod
    def build(
        cls,
        *,
        max_features_per_model: int,
    ) -> FeatureSetGrid:
        volume_options = {
            "volume zscore 10d": ["volume_zscore_10d"],
            "volume zscore 20d": ["volume_zscore_20d"],
            "volume zscore 60d": ["volume_zscore_60d"],
            "volume zscore 20d clipped": ["volume_zscore_20d_clip3"],
            "volume log1p zscore 20d": ["volume_log1p_zscore_20d"],
            "volume percentile rank 20d": ["volume_rank_20d"],
            "volume zscore 20d lag1": ["volume_zscore_20d_lag1"],
        }

        gdelt_options = {
            "GDELT zscore short": ["gdelt_sentiment_zscore_10d", "gdelt_article_count_zscore_6d"],
            "GDELT zscore medium": ["gdelt_sentiment_zscore_20d", "gdelt_article_count_zscore_20d"],
            "GDELT zscore short clipped": [
                "gdelt_sentiment_zscore_10d_clip3",
                "gdelt_article_count_zscore_6d_clip3",
            ],
            "GDELT sentiment zscore + log articles": [
                "gdelt_sentiment_zscore_10d",
                "gdelt_article_count_log1p_zscore_6d",
            ],
            "GDELT percentile rank 20d": ["gdelt_sentiment_rank_20d", "gdelt_article_count_rank_20d"],
            "GDELT zscore short + missing flag": [
                "gdelt_sentiment_zscore_10d",
                "gdelt_article_count_zscore_6d",
                "gdelt_sentiment_missing",
            ],
            "GDELT zscore short lag1": [
                "gdelt_sentiment_zscore_10d_lag1",
                "gdelt_article_count_zscore_6d_lag1",
            ],
        }

        gdelt_sentiment_options = {
            "GDELT sentiment zscore short": ["gdelt_sentiment_zscore_10d"],
            "GDELT sentiment zscore medium": ["gdelt_sentiment_zscore_20d"],
            "GDELT sentiment zscore short clipped": ["gdelt_sentiment_zscore_10d_clip3"],
            "GDELT sentiment percentile rank 20d": ["gdelt_sentiment_rank_20d"],
            "GDELT sentiment zscore short lag1-2": [
                "gdelt_sentiment_zscore_10d_lag1",
                "gdelt_sentiment_zscore_10d_lag2",
            ],
            "GDELT sentiment zscore short lag1-3": [
                "gdelt_sentiment_zscore_10d_lag1",
                "gdelt_sentiment_zscore_10d_lag2",
                "gdelt_sentiment_zscore_10d_lag3",
            ],
        }

        reddit_options = {
            "Reddit zscore short": ["reddit_sentiment_zscore_6d", "reddit_comment_count_zscore_6d"],
            "Reddit zscore medium": ["reddit_sentiment_zscore_20d", "reddit_comment_count_zscore_20d"],
            "Reddit zscore short clipped": [
                "reddit_sentiment_zscore_6d_clip3",
                "reddit_comment_count_zscore_6d_clip3",
            ],
            "Reddit sentiment zscore + log comments": [
                "reddit_sentiment_zscore_6d",
                "reddit_comment_count_log1p_zscore_6d",
            ],
            "Reddit percentile rank 20d": ["reddit_sentiment_rank_20d", "reddit_comment_count_rank_20d"],
            "Reddit zscore short + missing flag": [
                "reddit_sentiment_zscore_6d",
                "reddit_comment_count_zscore_6d",
                "reddit_sentiment_missing",
            ],
            "Reddit zscore short lag1": [
                "reddit_sentiment_zscore_6d_lag1",
                "reddit_comment_count_zscore_6d_lag1",
            ],
            "Reddit sentiment zscore short lag1-2": [
                "reddit_sentiment_zscore_6d_lag1",
                "reddit_sentiment_zscore_6d_lag2",
            ],
            "Reddit sentiment zscore short lag1-3": [
                "reddit_sentiment_zscore_6d_lag1",
                "reddit_sentiment_zscore_6d_lag2",
                "reddit_sentiment_zscore_6d_lag3",
            ],
        }

        gdelt_attention_options = {
            "GDELT attention zscore short": ["gdelt_article_count_zscore_6d"],
            "GDELT attention zscore medium": ["gdelt_article_count_zscore_20d"],
            "GDELT attention log1p zscore short": ["gdelt_article_count_log1p_zscore_6d"],
            "GDELT attention percentile rank 20d": ["gdelt_article_count_rank_20d"],
            "GDELT attention zscore short lag1-2": [
                "gdelt_article_count_zscore_6d_lag1",
                "gdelt_article_count_zscore_6d_lag2",
            ],
            "GDELT attention zscore short lag1-3": [
                "gdelt_article_count_zscore_6d_lag1",
                "gdelt_article_count_zscore_6d_lag2",
                "gdelt_article_count_zscore_6d_lag3",
            ],
        }

        reddit_attention_options = {
            "Reddit attention zscore short": ["reddit_comment_count_zscore_6d"],
            "Reddit attention zscore medium": ["reddit_comment_count_zscore_20d"],
            "Reddit attention log1p zscore short": ["reddit_comment_count_log1p_zscore_6d"],
            "Reddit attention percentile rank 20d": ["reddit_comment_count_rank_20d"],
            "Reddit attention zscore short lag1-2": [
                "reddit_comment_count_zscore_6d_lag1",
                "reddit_comment_count_zscore_6d_lag2",
            ],
            "Reddit attention zscore short lag1-3": [
                "reddit_comment_count_zscore_6d_lag1",
                "reddit_comment_count_zscore_6d_lag2",
                "reddit_comment_count_zscore_6d_lag3",
            ],
        }

        google_score_attention_options = {
            "Google score attention zscore 10d": ["google_trends_zscore_10d"],
            "Google score attention zscore 20d": ["google_trends_zscore_20d"],
            "Google score attention zscore 60d": ["google_trends_zscore_60d"],
            "Google score attention zscore 20d clipped": ["google_trends_zscore_20d_clip3"],
            "Google score attention percentile rank 20d": ["google_trends_rank_20d"],
            "Google score attention zscore 10d lag1-2": [
                "google_trends_zscore_10d_lag1",
                "google_trends_zscore_10d_lag2",
            ],
            "Google score attention zscore 10d lag1-3": [
                "google_trends_zscore_10d_lag1",
                "google_trends_zscore_10d_lag2",
                "google_trends_zscore_10d_lag3",
            ],
        }

        google_trends_options = {
            "Google train-median flag": ["google_trends_above_ticker_train_median"],
            "Google zscore 10d": ["google_trends_zscore_10d"],
            "Google zscore 20d": ["google_trends_zscore_20d"],
            "Google zscore 60d": ["google_trends_zscore_60d"],
            "Google zscore 20d clipped": ["google_trends_zscore_20d_clip3"],
            "Google percentile rank 20d": ["google_trends_rank_20d"],
            "Google zscore 10d lag1": ["google_trends_zscore_10d_lag1"],
        }

        base_volume_option = cls.BASE_VOLUME_OPTION
        baseline_feature_set = f"Model B - price + volume | {base_volume_option}"

        specs = cls._build_specs(
            volume_options=volume_options,
            gdelt_options=gdelt_options,
            reddit_options=reddit_options,
            google_trends_options=google_trends_options,
            gdelt_attention_options=gdelt_attention_options,
            reddit_attention_options=reddit_attention_options,
            google_score_attention_options=google_score_attention_options,
            base_volume_option=base_volume_option,
        )

        option_sources = {
            "volume": [volume_options],
            "gdelt": [gdelt_options, gdelt_sentiment_options, gdelt_attention_options],
            "reddit": [reddit_options, reddit_attention_options],
            "google": [google_trends_options, google_score_attention_options],
        }

        feature_sets: dict[str, list[str]] = {}
        metadata: dict[str, dict] = {}
        skipped: list[dict] = []

        for spec in specs:
            features = cls._features_from_spec(spec, option_sources)
            candidate = {**spec, "n_features": len(features), "features": features}
            if len(features) > max_features_per_model:
                skipped.append(candidate)
                continue
            feature_sets[spec["feature_set"]] = features
            metadata[spec["feature_set"]] = candidate

        feature_sets_to_test = list(feature_sets.keys())

        return FeatureSetGrid(
            price_features=list(cls.PRICE_FEATURES),
            volume_feature_options=volume_options,
            gdelt_feature_options=gdelt_options,
            gdelt_sentiment_feature_options=gdelt_sentiment_options,
            gdelt_attention_feature_options=gdelt_attention_options,
            reddit_feature_options=reddit_options,
            reddit_attention_feature_options=reddit_attention_options,
            google_trends_feature_options=google_trends_options,
            google_score_attention_feature_options=google_score_attention_options,
            derived_feature_columns={"google_trends_above_ticker_train_median"},
            base_volume_option=base_volume_option,
            baseline_feature_set=baseline_feature_set,
            feature_set_specs=specs,
            feature_sets=feature_sets,
            feature_set_metadata=metadata,
            skipped_feature_sets=skipped,
            feature_sets_to_test=feature_sets_to_test,
        )

    @classmethod
    def _feature_spec(
        cls,
        feature_set: str,
        feature_family: str,
        *,
        volume_option: str | None = None,
        gdelt_option: str | None = None,
        reddit_option: str | None = None,
        google_option: str | None = None,
    ) -> dict:
        return {
            "feature_set": feature_set,
            "feature_family": feature_family,
            "volume_option": volume_option,
            "gdelt_option": gdelt_option,
            "reddit_option": reddit_option,
            "google_option": google_option,
        }

    @classmethod
    def _build_specs(
        cls,
        *,
        volume_options: dict[str, list[str]],
        gdelt_options: dict[str, list[str]],
        reddit_options: dict[str, list[str]],
        google_trends_options: dict[str, list[str]],
        gdelt_attention_options: dict[str, list[str]],
        reddit_attention_options: dict[str, list[str]],
        google_score_attention_options: dict[str, list[str]],
        base_volume_option: str,
    ) -> list[dict]:
        spec = cls._feature_spec
        specs = [spec("Model A - price only", "price only")]

        for option_name in volume_options:
            specs.append(
                spec(
                    f"Model B - price + volume | {option_name}",
                    "price + volume",
                    volume_option=option_name,
                )
            )

        for option_name in gdelt_options:
            specs.append(
                spec(
                    f"Model C - price + volume + GDELT | {option_name}",
                    "price + volume + GDELT",
                    volume_option=base_volume_option,
                    gdelt_option=option_name,
                )
            )

        for option_name in reddit_options:
            specs.append(
                spec(
                    f"Model E - price + volume + Reddit | {option_name}",
                    "price + volume + Reddit",
                    volume_option=base_volume_option,
                    reddit_option=option_name,
                )
            )

        for option_name in google_trends_options:
            specs.append(
                spec(
                    f"Model G - price + volume + Google | {option_name}",
                    "price + volume + Google",
                    volume_option=base_volume_option,
                    google_option=option_name,
                )
            )

        for option_name in gdelt_attention_options:
            specs.append(
                spec(
                    f"Model J - price + volume + GDELT attention | {option_name}",
                    "price + volume + GDELT attention",
                    volume_option=base_volume_option,
                    gdelt_option=option_name,
                )
            )

        for option_name in reddit_attention_options:
            specs.append(
                spec(
                    f"Model K - price + volume + Reddit attention | {option_name}",
                    "price + volume + Reddit attention",
                    volume_option=base_volume_option,
                    reddit_option=option_name,
                )
            )

        for option_name in google_score_attention_options:
            specs.append(
                spec(
                    f"Model L - price + volume + Google score attention | {option_name}",
                    "price + volume + Google score attention",
                    volume_option=base_volume_option,
                    google_option=option_name,
                )
            )

        specs.extend(
            [
                spec(
                    "Model M - price + volume + GDELT + Reddit attention | zscore short",
                    "price + volume + GDELT + Reddit attention",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT attention zscore short",
                    reddit_option="Reddit attention zscore short",
                ),
                spec(
                    "Model M - price + volume + GDELT + Reddit attention | zscore medium",
                    "price + volume + GDELT + Reddit attention",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT attention zscore medium",
                    reddit_option="Reddit attention zscore medium",
                ),
                spec(
                    "Model N - price + volume + all attention | zscore short",
                    "price + volume + all attention",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT attention zscore short",
                    reddit_option="Reddit attention zscore short",
                    google_option="Google score attention zscore 10d",
                ),
                spec(
                    "Model N - price + volume + all attention | zscore medium",
                    "price + volume + all attention",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT attention zscore medium",
                    reddit_option="Reddit attention zscore medium",
                    google_option="Google score attention zscore 20d",
                ),
                spec(
                    "Model N - price + volume + all attention | percentile rank 20d",
                    "price + volume + all attention",
                    volume_option="volume percentile rank 20d",
                    gdelt_option="GDELT attention percentile rank 20d",
                    reddit_option="Reddit attention percentile rank 20d",
                    google_option="Google score attention percentile rank 20d",
                ),
                spec(
                    "Model D - price + volume + GDELT + Reddit | zscore short",
                    "price + volume + GDELT + Reddit",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT zscore short",
                    reddit_option="Reddit zscore short",
                ),
                spec(
                    "Model D - price + volume + GDELT + Reddit | zscore medium",
                    "price + volume + GDELT + Reddit",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT zscore medium",
                    reddit_option="Reddit zscore medium",
                ),
                spec(
                    "Model D - price + volume + GDELT + Reddit | clipped zscore short",
                    "price + volume + GDELT + Reddit",
                    volume_option="volume zscore 20d clipped",
                    gdelt_option="GDELT zscore short clipped",
                    reddit_option="Reddit zscore short clipped",
                ),
                spec(
                    "Model D - price + volume + GDELT + Reddit | lagged zscore short",
                    "price + volume + GDELT + Reddit",
                    volume_option="volume zscore 20d lag1",
                    gdelt_option="GDELT zscore short lag1",
                    reddit_option="Reddit zscore short lag1",
                ),
                spec(
                    "Model H - price + volume + GDELT + Google | zscore short",
                    "price + volume + GDELT + Google",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT zscore short",
                    google_option="Google zscore 10d",
                ),
                spec(
                    "Model H - price + volume + GDELT + Google | lagged zscore short",
                    "price + volume + GDELT + Google",
                    volume_option="volume zscore 20d lag1",
                    gdelt_option="GDELT zscore short lag1",
                    google_option="Google zscore 10d lag1",
                ),
                spec(
                    "Model I - price + volume + Reddit + Google | zscore short",
                    "price + volume + Reddit + Google",
                    volume_option=base_volume_option,
                    reddit_option="Reddit zscore short",
                    google_option="Google zscore 10d",
                ),
                spec(
                    "Model I - price + volume + Reddit + Google | lagged zscore short",
                    "price + volume + Reddit + Google",
                    volume_option="volume zscore 20d lag1",
                    reddit_option="Reddit zscore short lag1",
                    google_option="Google zscore 10d lag1",
                ),
                spec(
                    "Model F - price + volume + all alternative data | zscore short",
                    "price + volume + all alternative data",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT zscore short",
                    reddit_option="Reddit zscore short",
                    google_option="Google zscore 10d",
                ),
                spec(
                    "Model O - price + volume + GDELT sentiment + Reddit attention | zscore short",
                    "price + volume + GDELT sentiment + Reddit attention",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT sentiment zscore short",
                    reddit_option="Reddit attention zscore short",
                ),
                spec(
                    "Model O - price + volume + GDELT sentiment + Reddit attention | zscore medium",
                    "price + volume + GDELT sentiment + Reddit attention",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT sentiment zscore medium",
                    reddit_option="Reddit attention zscore medium",
                ),
                spec(
                    "Model O - price + volume + GDELT sentiment + Reddit attention | clipped sentiment + short attention",
                    "price + volume + GDELT sentiment + Reddit attention",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT sentiment zscore short clipped",
                    reddit_option="Reddit attention zscore short",
                ),
                spec(
                    "Model P - price + volume + GDELT sentiment + Reddit + Google attention | zscore short",
                    "price + volume + GDELT sentiment + Reddit + Google attention",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT sentiment zscore short",
                    reddit_option="Reddit attention zscore short",
                    google_option="Google score attention zscore 10d",
                ),
                spec(
                    "Model P - price + volume + GDELT sentiment + Reddit + Google attention | zscore medium",
                    "price + volume + GDELT sentiment + Reddit + Google attention",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT sentiment zscore medium",
                    reddit_option="Reddit attention zscore medium",
                    google_option="Google score attention zscore 20d",
                ),
                spec(
                    "Model Q - price + volume + GDELT sentiment lags | lag1-3",
                    "price + volume + GDELT sentiment lags",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT sentiment zscore short lag1-3",
                ),
                spec(
                    "Model R - price + volume + Reddit sentiment lags | lag1-3",
                    "price + volume + Reddit sentiment lags",
                    volume_option=base_volume_option,
                    reddit_option="Reddit sentiment zscore short lag1-3",
                ),
                spec(
                    "Model S - price + volume + GDELT attention lags | lag1-3",
                    "price + volume + GDELT attention lags",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT attention zscore short lag1-3",
                ),
                spec(
                    "Model T - price + volume + Reddit attention lags | lag1-3",
                    "price + volume + Reddit attention lags",
                    volume_option=base_volume_option,
                    reddit_option="Reddit attention zscore short lag1-3",
                ),
                spec(
                    "Model U - price + volume + Google attention lags | lag1-3",
                    "price + volume + Google attention lags",
                    volume_option=base_volume_option,
                    google_option="Google score attention zscore 10d lag1-3",
                ),
                spec(
                    "Model V - price + volume + GDELT sentiment lag1-3 + Reddit attention lag1-2",
                    "price + volume + GDELT sentiment lags + Reddit attention lags",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT sentiment zscore short lag1-3",
                    reddit_option="Reddit attention zscore short lag1-2",
                ),
                spec(
                    "Model V - price + volume + GDELT sentiment lag1-2 + Reddit attention lag1-3",
                    "price + volume + GDELT sentiment lags + Reddit attention lags",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT sentiment zscore short lag1-2",
                    reddit_option="Reddit attention zscore short lag1-3",
                ),
                spec(
                    "Model W - price + volume + Reddit sentiment lag1-3 + GDELT attention lag1-2",
                    "price + volume + Reddit sentiment lags + GDELT attention lags",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT attention zscore short lag1-2",
                    reddit_option="Reddit sentiment zscore short lag1-3",
                ),
                spec(
                    "Model W - price + volume + Reddit sentiment lag1-2 + GDELT attention lag1-3",
                    "price + volume + Reddit sentiment lags + GDELT attention lags",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT attention zscore short lag1-3",
                    reddit_option="Reddit sentiment zscore short lag1-2",
                ),
                spec(
                    "Model X - price + volume + GDELT sentiment lag1-2 + Google attention lag1-3",
                    "price + volume + GDELT sentiment lags + Google attention lags",
                    volume_option=base_volume_option,
                    gdelt_option="GDELT sentiment zscore short lag1-2",
                    google_option="Google score attention zscore 10d lag1-3",
                ),
                spec(
                    "Model Y - price + volume + Reddit sentiment lag1-2 + Google attention lag1-3",
                    "price + volume + Reddit sentiment lags + Google attention lags",
                    volume_option=base_volume_option,
                    reddit_option="Reddit sentiment zscore short lag1-2",
                    google_option="Google score attention zscore 10d lag1-3",
                ),
            ]
        )

        return specs

    @classmethod
    def _option_columns(cls, option_name: str, option_sources: list[dict[str, list[str]]]) -> list[str]:
        for source in option_sources:
            if option_name in source:
                return source[option_name]
        raise KeyError(f"Unknown feature option: {option_name}")

    @classmethod
    def _features_from_spec(cls, spec: dict, option_sources: dict[str, list[dict[str, list[str]]]]) -> list[str]:
        features = list(cls.PRICE_FEATURES)
        if spec["volume_option"] is not None:
            features.extend(cls._option_columns(spec["volume_option"], option_sources["volume"]))
        if spec["gdelt_option"] is not None:
            features.extend(cls._option_columns(spec["gdelt_option"], option_sources["gdelt"]))
        if spec["reddit_option"] is not None:
            features.extend(cls._option_columns(spec["reddit_option"], option_sources["reddit"]))
        if spec["google_option"] is not None:
            features.extend(cls._option_columns(spec["google_option"], option_sources["google"]))
        return features


class FeatureFrameBuilder:
    """Build the shared engineered feature frame used by model notebooks."""

    LAG_PERIODS = [1, 2, 3]
    LAG_SOURCE_COLUMNS = [
        "volume_zscore_20d",
        "gdelt_sentiment_zscore_10d",
        "gdelt_article_count_zscore_6d",
        "reddit_sentiment_zscore_6d",
        "reddit_comment_count_zscore_6d",
        "google_trends_zscore_10d",
    ]

    @staticmethod
    def transformed_source_series(
        frame: pd.DataFrame,
        source_col: str,
        value_transform: str | None = None,
    ) -> pd.Series:
        values = frame[source_col].astype(float)
        if value_transform is None:
            return values
        if value_transform == "log1p":
            return np.log1p(values.clip(lower=0.0))
        raise ValueError(f"Unsupported value_transform: {value_transform}")

    @classmethod
    def add_trailing_zscore(
        cls,
        frame: pd.DataFrame,
        source_col: str,
        out_col: str,
        window: int,
        *,
        min_periods: int | None = None,
        clip_value: float | None = None,
        value_transform: str | None = None,
    ) -> None:
        if min_periods is None:
            min_periods = max(3, window // 2)
        values = cls.transformed_source_series(frame, source_col, value_transform=value_transform)
        grouped = values.groupby(frame["ticker"])
        trailing_mean = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
        trailing_std = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).std())
        zscore = (values - trailing_mean) / trailing_std.replace(0.0, np.nan)
        if clip_value is not None:
            zscore = zscore.clip(-clip_value, clip_value)
        frame[out_col] = zscore

    @staticmethod
    def add_trailing_percentile_rank(
        frame: pd.DataFrame,
        source_col: str,
        out_col: str,
        window: int,
        *,
        min_periods: int | None = None,
    ) -> None:
        if min_periods is None:
            min_periods = max(3, window // 2)

        def trailing_rank(values: pd.Series) -> pd.Series:
            source_values = values.to_numpy(dtype=float)
            ranks = np.full(len(source_values), np.nan)

            for position, current_value in enumerate(source_values):
                start = max(0, position - window)
                historical_values = source_values[start:position]
                historical_values = historical_values[~np.isnan(historical_values)]
                if len(historical_values) < min_periods or np.isnan(current_value):
                    continue
                ranks[position] = (historical_values <= current_value).mean()

            return pd.Series(ranks, index=values.index)

        frame[out_col] = frame.groupby("ticker")[source_col].transform(trailing_rank)

    @staticmethod
    def add_group_lag(frame: pd.DataFrame, source_col: str, out_col: str, lag: int = 1) -> None:
        frame[out_col] = frame.groupby("ticker")[source_col].shift(lag)

    @classmethod
    def build_feature_frame(
        cls,
        raw_df: pd.DataFrame,
        neutral_band: float,
        *,
        include_lag_features: bool = True,
    ) -> pd.DataFrame:
        frame = raw_df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
        frame["reddit_sentiment_missing"] = frame["comm_reddit_vader_mean"].isna().astype(float)
        frame["gdelt_sentiment_missing"] = frame["gdelt_sentiment_score"].isna().astype(float)
        frame["comm_reddit_posts"] = frame["comm_reddit_posts"].fillna(0.0)
        frame["gdelt_articles"] = frame["gdelt_articles"].fillna(0.0)

        price_column = "close_stock_price" if "close_stock_price" in frame.columns else "stock_price"
        price_group = frame.groupby("ticker")[price_column]
        frame["return_1d"] = price_group.pct_change(1)
        frame["return_5d"] = price_group.pct_change(5)
        frame["return_20d"] = price_group.pct_change(20)
        frame["rolling_volatility_20d"] = (
            frame.groupby("ticker")["return_1d"].transform(lambda s: s.shift(1).rolling(20).std())
        )

        for window in [10, 20, 60]:
            cls.add_trailing_zscore(frame, "stock_volume", f"volume_zscore_{window}d", window)
        cls.add_trailing_zscore(frame, "stock_volume", "volume_zscore_20d_clip3", 20, clip_value=3.0)
        cls.add_trailing_zscore(frame, "stock_volume", "volume_log1p_zscore_20d", 20, value_transform="log1p")
        cls.add_trailing_percentile_rank(frame, "stock_volume", "volume_rank_20d", 20)

        for window in [10, 20]:
            cls.add_trailing_zscore(frame, "gdelt_sentiment_score", f"gdelt_sentiment_zscore_{window}d", window)
        for window in [6, 20]:
            cls.add_trailing_zscore(frame, "gdelt_articles", f"gdelt_article_count_zscore_{window}d", window)
            cls.add_trailing_zscore(
                frame,
                "gdelt_articles",
                f"gdelt_article_count_log1p_zscore_{window}d",
                window,
                value_transform="log1p",
            )
        cls.add_trailing_zscore(frame, "gdelt_sentiment_score", "gdelt_sentiment_zscore_10d_clip3", 10, clip_value=3.0)
        cls.add_trailing_zscore(frame, "gdelt_articles", "gdelt_article_count_zscore_6d_clip3", 6, clip_value=3.0)
        cls.add_trailing_percentile_rank(frame, "gdelt_sentiment_score", "gdelt_sentiment_rank_20d", 20)
        cls.add_trailing_percentile_rank(frame, "gdelt_articles", "gdelt_article_count_rank_20d", 20)

        for window in [6, 20]:
            cls.add_trailing_zscore(frame, "comm_reddit_vader_mean", f"reddit_sentiment_zscore_{window}d", window)
            cls.add_trailing_zscore(frame, "comm_reddit_posts", f"reddit_comment_count_zscore_{window}d", window)
            cls.add_trailing_zscore(
                frame,
                "comm_reddit_posts",
                f"reddit_comment_count_log1p_zscore_{window}d",
                window,
                value_transform="log1p",
            )
        cls.add_trailing_zscore(frame, "comm_reddit_vader_mean", "reddit_sentiment_zscore_6d_clip3", 6, clip_value=3.0)
        cls.add_trailing_zscore(frame, "comm_reddit_posts", "reddit_comment_count_zscore_6d_clip3", 6, clip_value=3.0)
        cls.add_trailing_percentile_rank(frame, "comm_reddit_vader_mean", "reddit_sentiment_rank_20d", 20)
        cls.add_trailing_percentile_rank(frame, "comm_reddit_posts", "reddit_comment_count_rank_20d", 20)

        for window in [10, 20, 60]:
            cls.add_trailing_zscore(frame, "google_trends_score", f"google_trends_zscore_{window}d", window)
        cls.add_trailing_zscore(frame, "google_trends_score", "google_trends_zscore_20d_clip3", 20, clip_value=3.0)
        cls.add_trailing_percentile_rank(frame, "google_trends_score", "google_trends_rank_20d", 20)

        if include_lag_features:
            for source_col in cls.LAG_SOURCE_COLUMNS:
                for lag in cls.LAG_PERIODS:
                    cls.add_group_lag(frame, source_col, f"{source_col}_lag{lag}", lag=lag)

        frame["future_return_1d"] = price_group.shift(-1) / frame[price_column] - 1.0
        frame["target"] = np.select(
            [
                frame["future_return_1d"] < -neutral_band,
                frame["future_return_1d"] > neutral_band,
            ],
            [0, 1],
            default=np.nan,
        )
        frame["target_available"] = frame["future_return_1d"].notna()
        frame["is_neutral"] = frame["target_available"] & frame["future_return_1d"].abs().le(neutral_band)
        return frame

    @staticmethod
    def add_google_trends_train_median_feature(
        train_input_df: pd.DataFrame,
        eval_input_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_output_df = train_input_df.copy()
        eval_output_df = eval_input_df.copy()
        median_by_ticker = train_output_df.groupby("ticker")["google_trends_score"].median()
        global_median = train_output_df["google_trends_score"].median()

        def add_flag(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            ticker_medians = frame["ticker"].map(median_by_ticker).fillna(global_median)
            frame["google_trends_above_ticker_train_median"] = (
                frame["google_trends_score"] > ticker_medians
            ).astype(float)
            return frame

        return add_flag(train_output_df), add_flag(eval_output_df)
