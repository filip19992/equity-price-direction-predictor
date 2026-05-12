from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from notebook_utils.metrics import METRIC_COLUMNS


class ModelReportBuilder:
    """Build unified validation and test reports for model notebooks."""

    @staticmethod
    def available_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
        return [column for column in columns if column in frame.columns]

    @staticmethod
    def metric_sort_columns(selection_metric: str = "balanced_accuracy") -> list[str]:
        return list(dict.fromkeys([selection_metric, "balanced_accuracy", "f1_score", "accuracy"]))

    @staticmethod
    def save_final_test_verification(
        report_df: pd.DataFrame,
        *,
        model_name: str,
        output_dir: str | Path,
    ) -> Path:
        """Save the final validation-selected family test report for thesis comparison."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / f"{model_name}_final_test_verification.csv"
        report_df.to_csv(report_path, index=False)
        return report_path

    @classmethod
    def select_best_validation_by_feature_set(
        cls,
        validation_results_df: pd.DataFrame,
        *,
        selection_metric: str = "balanced_accuracy",
    ) -> pd.DataFrame:
        """Select one validation-winning hyperparameter row per feature set."""
        sort_metrics = cls.metric_sort_columns(selection_metric)
        required_columns = {"feature_set", *sort_metrics}
        missing_columns = sorted(required_columns.difference(validation_results_df.columns))
        if missing_columns:
            raise KeyError(f"Validation results are missing required columns: {missing_columns}")

        return (
            validation_results_df.sort_values(
                ["feature_set", *sort_metrics, "param_set"],
                ascending=[True, *([False] * len(sort_metrics)), True],
            )
            .groupby("feature_set", as_index=False)
            .head(1)
            .sort_values([*sort_metrics, "feature_set"], ascending=[*([False] * len(sort_metrics)), True])
            .reset_index(drop=True)
        )

    @classmethod
    def build_validation_best_by_feature_set_report(
        cls,
        validation_best_by_feature_set_df: pd.DataFrame,
        *,
        param_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        """Validation report: best model per feature set, sorted by balanced accuracy."""
        report = validation_best_by_feature_set_df.rename(
            columns={metric: f"validation_{metric}" for metric in METRIC_COLUMNS}
        )
        columns = [
            "feature_family",
            "feature_set",
            "n_features",
            "param_set",
            *param_columns,
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_f1_score",
        ]
        return (
            report[cls.available_columns(report, columns)]
            .sort_values(
                ["validation_balanced_accuracy", "validation_f1_score", "validation_accuracy", "feature_set"],
                ascending=[False, False, False, True],
            )
            .reset_index(drop=True)
        )

    @classmethod
    def build_simple_hyperparameter_summary(
        cls,
        *,
        best_validation_params_df: pd.DataFrame,
        threshold_calibration_results_df: pd.DataFrame,
        test_best_validation_params_df: pd.DataFrame,
        baseline_feature_set: str,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Combine validation-selected params, calibration, and test metrics."""
        baseline_walk_forward_row = best_validation_params_df[
            best_validation_params_df["feature_set"].eq(baseline_feature_set)
        ].iloc[0]
        baseline_calibration_row = threshold_calibration_results_df[
            threshold_calibration_results_df["feature_set"].eq(baseline_feature_set)
        ].iloc[0]
        baseline_test_row = test_best_validation_params_df[
            test_best_validation_params_df["feature_set"].eq(baseline_feature_set)
        ].iloc[0]

        summary = best_validation_params_df.rename(
            columns={
                "accuracy": "walk_forward_mean_accuracy",
                "balanced_accuracy": "walk_forward_mean_balanced_accuracy",
                "f1_score": "walk_forward_mean_f1_score",
            }
        )

        summary = summary.merge(
            threshold_calibration_results_df[["feature_set", "accuracy", "balanced_accuracy", "f1_score"]].rename(
                columns={
                    "accuracy": "threshold_calibration_accuracy",
                    "balanced_accuracy": "threshold_calibration_balanced_accuracy",
                    "f1_score": "threshold_calibration_f1_score",
                }
            ),
            on="feature_set",
            how="left",
        ).merge(
            test_best_validation_params_df[["feature_set", "accuracy", "balanced_accuracy", "f1_score"]].rename(
                columns={
                    "accuracy": "test_accuracy",
                    "balanced_accuracy": "test_balanced_accuracy",
                    "f1_score": "test_f1_score",
                }
            ),
            on="feature_set",
            how="left",
        )

        return summary, baseline_walk_forward_row, baseline_calibration_row, baseline_test_row

    @classmethod
    def build_validation_selected_family_test_report(
        cls,
        simple_summary: pd.DataFrame,
        *,
        baseline_family: str = "price + volume",
        param_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        """Test report for the best validation-selected model in each feature family."""
        report = cls._with_price_volume_test_lift(simple_summary, baseline_family=baseline_family)
        report = cls._rename_validation_columns(report)
        family_winners = (
            report.sort_values(
                [
                    "feature_family",
                    "validation_balanced_accuracy",
                    "validation_f1_score",
                    "validation_accuracy",
                    "feature_set",
                ],
                ascending=[True, False, False, False, True],
            )
            .groupby("feature_family", as_index=False)
            .head(1)
            .sort_values(
                [
                    "test_balanced_accuracy",
                    "test_f1_score",
                    "test_accuracy",
                    "validation_balanced_accuracy",
                    "feature_family",
                ],
                ascending=[False, False, False, False, True],
            )
            .reset_index(drop=True)
        )
        columns = [
            "feature_family",
            "feature_set",
            "n_features",
            "param_set",
            *param_columns,
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_f1_score",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1_score",
            "price_volume_baseline_feature_set",
            "price_volume_baseline_test_balanced_accuracy",
            "test_balanced_accuracy_change_vs_price_volume",
        ]
        return family_winners[cls.available_columns(family_winners, columns)]

    @classmethod
    def _rename_validation_columns(cls, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.rename(
            columns={
                "walk_forward_mean_accuracy": "validation_accuracy",
                "walk_forward_mean_balanced_accuracy": "validation_balanced_accuracy",
                "walk_forward_mean_f1_score": "validation_f1_score",
            }
        )

    @classmethod
    def _with_price_volume_test_lift(
        cls,
        simple_summary: pd.DataFrame,
        *,
        baseline_family: str,
    ) -> pd.DataFrame:
        baseline_candidates = simple_summary[simple_summary["feature_family"].eq(baseline_family)]
        if baseline_candidates.empty:
            raise ValueError(f"No baseline feature family found: {baseline_family}")

        baseline_row = (
            baseline_candidates.sort_values(
                [
                    "walk_forward_mean_balanced_accuracy",
                    "walk_forward_mean_f1_score",
                    "walk_forward_mean_accuracy",
                    "feature_set",
                ],
                ascending=[False, False, False, True],
            )
            .head(1)
            .iloc[0]
        )
        report = simple_summary.copy()
        report["price_volume_baseline_feature_set"] = baseline_row["feature_set"]
        report["price_volume_baseline_test_balanced_accuracy"] = baseline_row["test_balanced_accuracy"]
        report["test_balanced_accuracy_change_vs_price_volume"] = (
            report["test_balanced_accuracy"] - baseline_row["test_balanced_accuracy"]
        )
        return report
