from __future__ import annotations

from typing import Iterable

import pandas as pd


class ModelReportBuilder:
    """Build common validation/test comparison tables."""

    @staticmethod
    def available_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
        return [column for column in columns if column in frame.columns]

    @classmethod
    def build_simple_hyperparameter_summary(
        cls,
        *,
        best_validation_params_df: pd.DataFrame,
        threshold_calibration_results_df: pd.DataFrame,
        test_best_validation_params_df: pd.DataFrame,
        baseline_feature_set: str,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        baseline_walk_forward_row = best_validation_params_df[
            best_validation_params_df["feature_set"].eq(baseline_feature_set)
        ].iloc[0]
        baseline_calibration_row = threshold_calibration_results_df[
            threshold_calibration_results_df["feature_set"].eq(baseline_feature_set)
        ].iloc[0]
        baseline_test_row = test_best_validation_params_df[
            test_best_validation_params_df["feature_set"].eq(baseline_feature_set)
        ].iloc[0]

        simple_summary = best_validation_params_df.rename(
            columns={
                "selection_score": "walk_forward_selection_score",
                "balanced_accuracy": "walk_forward_mean_balanced_accuracy",
                "balanced_accuracy_std": "walk_forward_std_balanced_accuracy",
                "balanced_accuracy_min": "walk_forward_min_balanced_accuracy",
                "accuracy": "walk_forward_mean_accuracy",
                "f1_score": "walk_forward_mean_f1_score",
                "predicted_positive_rate": "walk_forward_mean_predicted_positive_rate",
            }
        )

        simple_summary = simple_summary.merge(
            threshold_calibration_results_df[
                ["feature_set", "accuracy", "balanced_accuracy", "f1_score", "predicted_positive_rate", "decision_threshold"]
            ].rename(
                columns={
                    "accuracy": "threshold_calibration_accuracy",
                    "balanced_accuracy": "threshold_calibration_balanced_accuracy",
                    "f1_score": "threshold_calibration_f1_score",
                    "predicted_positive_rate": "threshold_calibration_predicted_positive_rate",
                    "decision_threshold": "threshold_calibration_decision_threshold",
                }
            ),
            on="feature_set",
            how="left",
        ).merge(
            test_best_validation_params_df[
                ["feature_set", "accuracy", "balanced_accuracy", "f1_score", "predicted_positive_rate", "decision_threshold"]
            ].rename(
                columns={
                    "accuracy": "test_accuracy",
                    "balanced_accuracy": "test_balanced_accuracy",
                    "f1_score": "test_f1_score",
                    "predicted_positive_rate": "test_predicted_positive_rate",
                    "decision_threshold": "test_decision_threshold",
                }
            ),
            on="feature_set",
            how="left",
        )

        simple_summary["walk_forward_selection_score_lift_vs_baseline"] = (
            simple_summary["walk_forward_selection_score"] - baseline_walk_forward_row["selection_score"]
        )
        simple_summary["walk_forward_balanced_accuracy_lift_vs_baseline"] = (
            simple_summary["walk_forward_mean_balanced_accuracy"] - baseline_walk_forward_row["balanced_accuracy"]
        )
        simple_summary["threshold_calibration_balanced_accuracy_lift_vs_baseline"] = (
            simple_summary["threshold_calibration_balanced_accuracy"] - baseline_calibration_row["balanced_accuracy"]
        )
        simple_summary["threshold_calibration_f1_score_lift_vs_baseline"] = (
            simple_summary["threshold_calibration_f1_score"] - baseline_calibration_row["f1_score"]
        )
        simple_summary["test_accuracy_lift_vs_baseline"] = (
            simple_summary["test_accuracy"] - baseline_test_row["accuracy"]
        )
        simple_summary["test_balanced_accuracy_lift_vs_baseline"] = (
            simple_summary["test_balanced_accuracy"] - baseline_test_row["balanced_accuracy"]
        )
        simple_summary["test_f1_score_lift_vs_baseline"] = (
            simple_summary["test_f1_score"] - baseline_test_row["f1_score"]
        )

        return simple_summary, baseline_walk_forward_row, baseline_calibration_row, baseline_test_row

    @classmethod
    def simple_summary_columns(cls, param_columns: Iterable[str] = (), extra_columns: Iterable[str] = ()) -> list[str]:
        return [
            "feature_set",
            "feature_family",
            "n_features",
            "param_set",
            *param_columns,
            "walk_forward_selection_score",
            "walk_forward_mean_balanced_accuracy",
            "walk_forward_std_balanced_accuracy",
            "walk_forward_min_balanced_accuracy",
            "walk_forward_balanced_accuracy_lift_vs_baseline",
            "threshold_calibration_decision_threshold",
            "threshold_calibration_balanced_accuracy",
            "threshold_calibration_f1_score",
            "test_balanced_accuracy",
            "test_accuracy",
            "test_f1_score",
            "test_balanced_accuracy_lift_vs_baseline",
            "test_f1_score_lift_vs_baseline",
            *extra_columns,
            "features",
        ]

    @classmethod
    def build_alternative_data_research_question(
        cls,
        simple_summary: pd.DataFrame,
        *,
        baseline_family: str = "price + volume",
    ) -> pd.DataFrame:
        columns = [
            "comparison_group",
            "feature_family",
            "feature_set",
            "n_features",
            "param_set",
            "walk_forward_mean_balanced_accuracy",
            "threshold_calibration_balanced_accuracy",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1_score",
            "test_accuracy_lift_vs_price_volume",
            "test_balanced_accuracy_lift_vs_price_volume",
            "test_f1_score_lift_vs_price_volume",
            "improves_test_accuracy_vs_price_volume",
            "improves_test_balanced_accuracy_vs_price_volume",
        ]

        baseline_row = (
            simple_summary[simple_summary["feature_family"].eq(baseline_family)]
            .sort_values(
                ["walk_forward_selection_score", "walk_forward_mean_balanced_accuracy", "threshold_calibration_balanced_accuracy"],
                ascending=False,
            )
            .head(1)
            .iloc[0]
        )

        comparison = simple_summary.copy()
        comparison["uses_alternative_data"] = ~comparison["feature_family"].isin(["price only", baseline_family])
        comparison["comparison_baseline_feature_set"] = baseline_row["feature_set"]
        comparison["test_accuracy_lift_vs_price_volume"] = comparison["test_accuracy"] - baseline_row["test_accuracy"]
        comparison["test_balanced_accuracy_lift_vs_price_volume"] = (
            comparison["test_balanced_accuracy"] - baseline_row["test_balanced_accuracy"]
        )
        comparison["test_f1_score_lift_vs_price_volume"] = comparison["test_f1_score"] - baseline_row["test_f1_score"]
        comparison["walk_forward_balanced_accuracy_lift_vs_price_volume"] = (
            comparison["walk_forward_mean_balanced_accuracy"] - baseline_row["walk_forward_mean_balanced_accuracy"]
        )
        comparison["threshold_calibration_balanced_accuracy_lift_vs_price_volume"] = (
            comparison["threshold_calibration_balanced_accuracy"] - baseline_row["threshold_calibration_balanced_accuracy"]
        )

        best_alternative_by_family = (
            comparison[comparison["uses_alternative_data"]]
            .sort_values(
                [
                    "feature_family",
                    "walk_forward_selection_score",
                    "walk_forward_mean_balanced_accuracy",
                    "threshold_calibration_balanced_accuracy",
                    "test_balanced_accuracy",
                ],
                ascending=[True, False, False, False, False],
            )
            .groupby("feature_family", as_index=False)
            .head(1)
            .sort_values(
                ["test_accuracy_lift_vs_price_volume", "test_balanced_accuracy_lift_vs_price_volume", "test_f1_score_lift_vs_price_volume"],
                ascending=False,
            )
            .reset_index(drop=True)
        )

        baseline_frame = pd.DataFrame(
            [
                {
                    "comparison_group": "baseline",
                    "feature_family": baseline_row["feature_family"],
                    "feature_set": baseline_row["feature_set"],
                    "n_features": baseline_row["n_features"],
                    "param_set": baseline_row["param_set"],
                    "walk_forward_mean_balanced_accuracy": baseline_row["walk_forward_mean_balanced_accuracy"],
                    "threshold_calibration_balanced_accuracy": baseline_row["threshold_calibration_balanced_accuracy"],
                    "test_accuracy": baseline_row["test_accuracy"],
                    "test_balanced_accuracy": baseline_row["test_balanced_accuracy"],
                    "test_f1_score": baseline_row["test_f1_score"],
                    "test_accuracy_lift_vs_price_volume": 0.0,
                    "test_balanced_accuracy_lift_vs_price_volume": 0.0,
                    "test_f1_score_lift_vs_price_volume": 0.0,
                    "improves_test_accuracy_vs_price_volume": False,
                    "improves_test_balanced_accuracy_vs_price_volume": False,
                }
            ]
        )

        alternative_frame = best_alternative_by_family.copy()
        alternative_frame["comparison_group"] = "alternative data"
        alternative_frame["improves_test_accuracy_vs_price_volume"] = (
            alternative_frame["test_accuracy_lift_vs_price_volume"] > 0
        )
        alternative_frame["improves_test_balanced_accuracy_vs_price_volume"] = (
            alternative_frame["test_balanced_accuracy_lift_vs_price_volume"] > 0
        )

        return pd.concat(
            [baseline_frame[columns], alternative_frame[columns]],
            ignore_index=True,
        )

    @classmethod
    def build_attention_research_question(
        cls,
        simple_summary: pd.DataFrame,
        *,
        param_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        attention_summary = simple_summary[
            simple_summary["feature_family"].str.contains("attention", case=False, na=False)
        ].copy()
        best_attention_by_family = (
            attention_summary.sort_values(
                [
                    "feature_family",
                    "walk_forward_selection_score",
                    "walk_forward_mean_balanced_accuracy",
                    "threshold_calibration_balanced_accuracy",
                    "test_balanced_accuracy",
                ],
                ascending=[True, False, False, False, False],
            )
            .groupby("feature_family", as_index=False)
            .head(1)
            .sort_values(
                ["test_accuracy_lift_vs_baseline", "test_balanced_accuracy_lift_vs_baseline", "test_f1_score_lift_vs_baseline"],
                ascending=False,
            )
            .reset_index(drop=True)
        )
        columns = [
            "feature_family",
            "feature_set",
            "n_features",
            "param_set",
            *param_columns,
            "walk_forward_mean_balanced_accuracy",
            "threshold_calibration_balanced_accuracy",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1_score",
            "test_accuracy_lift_vs_baseline",
            "test_balanced_accuracy_lift_vs_baseline",
            "test_f1_score_lift_vs_baseline",
            "features",
        ]
        return best_attention_by_family[cls.available_columns(best_attention_by_family, columns)]

    @classmethod
    def build_best_model_by_family(
        cls,
        simple_summary: pd.DataFrame,
        *,
        sort_by_accuracy_first: bool = False,
    ) -> pd.DataFrame:
        final_sort_columns = ["test_balanced_accuracy", "test_f1_score"]
        if sort_by_accuracy_first:
            final_sort_columns = ["test_accuracy", "test_balanced_accuracy", "test_f1_score"]

        return (
            simple_summary.sort_values(
                ["feature_family", "test_accuracy", "test_balanced_accuracy", "test_f1_score", "walk_forward_selection_score"],
                ascending=[True, False, False, False, False],
            )
            .groupby("feature_family", as_index=False)
            .head(1)
            .sort_values(final_sort_columns, ascending=False)
            .reset_index(drop=True)
        )

    @classmethod
    def family_comparison_columns(
        cls,
        *,
        param_columns: Iterable[str] = (),
        extra_columns: Iterable[str] = (),
        compact: bool = False,
    ) -> list[str]:
        if compact:
            return [
                "feature_family",
                "feature_set",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_f1_score",
                "test_predicted_positive_rate",
            ]
        return [
            "feature_family",
            "feature_set",
            "n_features",
            "param_set",
            *param_columns,
            "walk_forward_selection_score",
            "walk_forward_mean_balanced_accuracy",
            "walk_forward_std_balanced_accuracy",
            "threshold_calibration_decision_threshold",
            "threshold_calibration_balanced_accuracy",
            "threshold_calibration_f1_score",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1_score",
            "test_predicted_positive_rate",
            *extra_columns,
            "features",
        ]
