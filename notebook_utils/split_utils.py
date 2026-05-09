from __future__ import annotations

import numpy as np
import pandas as pd


def make_split_dates(
    frame: pd.DataFrame,
    test_size: float,
    validation_fraction_within_pretest: float,
    min_validation_dates: int,
    gap_days: int,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]:
    unique_dates = sorted(frame["date"].drop_duplicates())

    test_start_idx = int(np.floor(len(unique_dates) * (1.0 - test_size)))
    test_start_idx = min(max(test_start_idx, 2), len(unique_dates) - 1)
    pretest_end_idx = max(test_start_idx - gap_days, 1)
    pretest_dates = unique_dates[:pretest_end_idx]

    validation_size = int(np.floor(len(pretest_dates) * validation_fraction_within_pretest))
    validation_size = max(min_validation_dates, validation_size)
    validation_size = min(max(validation_size, 1), len(pretest_dates) - 1)

    validation_start_idx = len(pretest_dates) - validation_size
    train_end_idx = max(validation_start_idx - gap_days, 1)

    train_dates = unique_dates[:train_end_idx]
    validation_dates = pretest_dates[validation_start_idx:]
    test_dates = unique_dates[test_start_idx:]
    return train_dates, validation_dates, test_dates


def make_calendar_year_split_dates(
    frame: pd.DataFrame,
    *,
    train_start_date: str,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    test_start_date: str,
    test_end_date: str,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]:
    unique_dates = pd.Series(sorted(frame["date"].drop_duplicates()))

    def dates_between(start_date: str, end_date: str, split_name: str) -> list[pd.Timestamp]:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        selected_dates = unique_dates[(unique_dates >= start) & (unique_dates <= end)].tolist()
        if not selected_dates:
            raise ValueError(f"No session dates found for {split_name}: {start_date} to {end_date}")
        return selected_dates

    train_dates = dates_between(train_start_date, train_end_date, "train")
    validation_dates = dates_between(validation_start_date, validation_end_date, "validation")
    test_dates = dates_between(test_start_date, test_end_date, "test")

    split_sets = {
        "train": set(train_dates),
        "validation": set(validation_dates),
        "test": set(test_dates),
    }
    if (
        split_sets["train"] & split_sets["validation"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["validation"] & split_sets["test"]
    ):
        raise ValueError("Calendar split date ranges overlap.")
    if not (max(train_dates) < min(validation_dates) < max(validation_dates) < min(test_dates)):
        raise ValueError("Calendar split must be ordered as train < validation < test.")

    return train_dates, validation_dates, test_dates


def make_configured_split_dates(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]:
    split_strategy = config.get("split_strategy", "fractional")
    if split_strategy == "calendar_year":
        return make_calendar_year_split_dates(
            frame,
            train_start_date=config["train_start_date"],
            train_end_date=config["train_end_date"],
            validation_start_date=config["validation_start_date"],
            validation_end_date=config["validation_end_date"],
            test_start_date=config["test_start_date"],
            test_end_date=config["test_end_date"],
        )
    if split_strategy == "fractional":
        return make_split_dates(
            frame,
            test_size=config["test_size"],
            validation_fraction_within_pretest=config["validation_fraction_within_pretest"],
            min_validation_dates=config["min_validation_dates"],
            gap_days=config["gap_days"],
        )
    raise ValueError(f"Unsupported split_strategy: {split_strategy}")


def subset_by_dates(frame: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    return frame[frame["date"].isin(dates)].copy()


def make_train_validation_test_split_by_date(
    frame: pd.DataFrame,
    test_size: float,
    validation_fraction_within_pretest: float,
    min_validation_dates: int,
    gap_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dates, validation_dates, test_dates = make_split_dates(
        frame,
        test_size=test_size,
        validation_fraction_within_pretest=validation_fraction_within_pretest,
        min_validation_dates=min_validation_dates,
        gap_days=gap_days,
    )
    return subset_by_dates(frame, train_dates), subset_by_dates(frame, validation_dates), subset_by_dates(frame, test_dates)


def make_walk_forward_fold_specs(
    session_dates: list[pd.Timestamp],
    *,
    n_folds: int,
    validation_size: int,
    min_train_dates: int,
    gap_days: int,
) -> list[dict]:
    dates = sorted(pd.Series(session_dates).drop_duplicates())
    if len(dates) == 0:
        raise ValueError("No session dates available for walk-forward validation.")

    max_validation_size = (len(dates) - min_train_dates - gap_days) // n_folds
    if max_validation_size < 1:
        raise ValueError(
            "Not enough dates for requested walk-forward folds. "
            f"dates={len(dates)}, n_folds={n_folds}, min_train_dates={min_train_dates}, gap_days={gap_days}"
        )
    validation_size = min(validation_size, max_validation_size)
    first_validation_start = len(dates) - n_folds * validation_size

    fold_specs = []
    for fold_idx in range(n_folds):
        validation_start = first_validation_start + fold_idx * validation_size
        validation_end = validation_start + validation_size
        train_end = validation_start - gap_days
        train_fold_dates = dates[:train_end]
        validation_fold_dates = dates[validation_start:validation_end]
        if len(train_fold_dates) < min_train_dates or not validation_fold_dates:
            continue
        fold_specs.append(
            {
                "fold": fold_idx + 1,
                "train_dates": train_fold_dates,
                "validation_dates": validation_fold_dates,
                "train_date_min": min(train_fold_dates),
                "train_date_max": max(train_fold_dates),
                "validation_date_min": min(validation_fold_dates),
                "validation_date_max": max(validation_fold_dates),
                "train_n_dates": len(train_fold_dates),
                "validation_n_dates": len(validation_fold_dates),
            }
        )

    if len(fold_specs) != n_folds:
        raise ValueError(f"Created {len(fold_specs)} walk-forward folds, expected {n_folds}.")
    return fold_specs
