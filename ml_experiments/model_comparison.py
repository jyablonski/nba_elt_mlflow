"""
Model comparison and benchmarking utilities.

Provides comprehensive tools for comparing models, running time-series validaton,
and selecting the best model for production deployment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    cross_validate,
    TimeSeriesSplit,
)
from scipy import stats

from ml_experiments.models import ModelFactory
from ml_experiments.evaluation import (
    ModelEvaluator,
    EvaluationMetrics,
)


@dataclass
class ComparisonResult:
    """Results from model comparison."""

    rankings: pd.DataFrame
    detailed_metrics: Dict[str, EvaluationMetrics]
    statistical_tests: pd.DataFrame
    best_model_name: str
    best_model: BaseEstimator
    improvement_over_baseline: float


class ModelComparison:
    """
    Comprehensive model comparison and benchmarking engine.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the model comparison framework.
        """
        self.random_state = random_state
        self.factory = ModelFactory()
        self.evaluator = ModelEvaluator()
        self.comparison_results: Dict[str, Dict[str, Any]] = {}

    def compare_models(
        self,
        models: Dict[str, BaseEstimator],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_splits: int = 5,
        use_time_series_split: bool = True,
    ) -> ComparisonResult:
        """
        Compare multiple models using TimeSeries Cross-Validation.

        Args:
            models: Dictionary of model_name -> model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            cv_splits: Number of cross-validation splits
            use_time_series_split: If True, uses TimeSeriesSplit (No shuffling).
                                   Crucial for sports data to prevent leakage.
        """

        # 1. Setup Cross-Validation Strategy
        if use_time_series_split:
            # Best Practice: Train on past, validate on future
            cv = TimeSeriesSplit(n_splits=cv_splits)
        else:
            # Only use StratifiedKFold if data is NOT time-dependent
            from sklearn.model_selection import StratifiedKFold

            cv = StratifiedKFold(
                n_splits=cv_splits, shuffle=True, random_state=self.random_state
            )

        results_list = []
        cv_scores_storage = {}
        detailed_metrics = {}

        # 2. Iterate Models
        for name, model in models.items():
            print(f"Evaluating {name}...")

            # Clone to ensure fresh start
            model_clone = clone(model)

            # --- A. Efficient Cross-Validation ---
            # Best Practice: Use cross_validate to compute all metrics in one pass
            scoring = ["accuracy", "roc_auc", "f1", "neg_log_loss", "neg_brier_score"]
            cv_results = cross_validate(
                model_clone, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
            )

            # Store raw scores for statistical testing later
            cv_scores_storage[name] = {
                "accuracy": cv_results["test_accuracy"],
                "roc_auc": cv_results["test_roc_auc"],
                "f1": cv_results["test_f1"],
            }

            # --- B. Test Set Evaluation ---
            # Retrain on FULL training set before testing
            model_clone.fit(X_train, y_train)

            metrics = self.evaluator.evaluate(
                model_clone,
                X_test,
                y_test,
                model_name=name,
                cv_scores=cv_results["test_accuracy"],
            )
            detailed_metrics[name] = metrics

            self.comparison_results[name] = {
                "model": model_clone,
                "metrics": metrics,
            }

            # --- C. Aggregate Results ---
            results_list.append(
                {
                    "model": name,
                    "cv_auc_mean": cv_results["test_roc_auc"].mean(),
                    "cv_auc_std": cv_results["test_roc_auc"].std(),
                    "cv_acc_mean": cv_results["test_accuracy"].mean(),
                    "test_auc": metrics.auc,
                    "test_acc": metrics.accuracy,
                    "test_mcc": metrics.mcc,
                    "test_brier": metrics.brier_score,
                    "calibration_ece": metrics.calibration_error,
                }
            )

        # 3. Compile Rankings
        rankings = pd.DataFrame(results_list)
        rankings = rankings.sort_values("cv_auc_mean", ascending=False).reset_index(
            drop=True
        )
        rankings["rank"] = rankings.index + 1

        # 4. Run Statistical Tests (Best vs Others)
        stat_tests = self._run_statistical_tests(cv_scores_storage)

        # 5. Identify Best Model
        best_name = rankings.iloc[0]["model"]
        best_model = self.comparison_results[best_name]["model"]

        # Calculate improvement (vs random baseline of 0.5 AUC)
        baseline_auc = 0.5
        best_auc = rankings.iloc[0]["test_auc"]
        improvement = (best_auc - baseline_auc) / baseline_auc * 100

        return ComparisonResult(
            rankings=rankings,
            detailed_metrics=detailed_metrics,
            statistical_tests=stat_tests,
            best_model_name=best_name,
            best_model=best_model,
            improvement_over_baseline=improvement,
        )

    def _run_statistical_tests(
        self,
        cv_scores_dict: Dict[str, Dict[str, np.ndarray]],
    ) -> pd.DataFrame:
        """Run pairwise Paired T-Tests between models."""
        model_names = list(cv_scores_dict.keys())
        test_results = []

        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1 :]:
                # Compare AUC scores across folds
                scores_a = cv_scores_dict[model_a]["roc_auc"]
                scores_b = cv_scores_dict[model_b]["roc_auc"]

                # Paired t-test
                t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

                test_results.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_diff": scores_a.mean() - scores_b.mean(),
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }
                )

        return pd.DataFrame(test_results)

    def analyze_feature_stability(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        n_iterations: int = 10,
    ) -> pd.DataFrame:
        """
        Analyze feature importance stability via Bootstrapping.
        Handles both Tree (feature_importances_) and Linear (coef_) models.
        """
        importance_matrix = []
        feature_names = X.columns.tolist()

        for i in range(n_iterations):
            # Bootstrap sample
            rng = np.random.default_rng(self.random_state + i)
            indices = rng.choice(len(X), size=len(X), replace=True)

            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]

            # Clone and fit
            model_clone = clone(model)
            model_clone.fit(X_boot, y_boot)

            # Extract Importance
            imp = self._extract_importance(model_clone)
            if imp is not None:
                importance_matrix.append(imp)

        if not importance_matrix:
            return pd.DataFrame()

        importance_array = np.array(importance_matrix)

        return pd.DataFrame(
            {
                "feature": feature_names,
                "mean": importance_array.mean(axis=0),
                "std": importance_array.std(axis=0),
                # Coefficient of Variation (lower is more stable)
                "cv": importance_array.std(axis=0)
                / (np.abs(importance_array.mean(axis=0)) + 1e-9),
            }
        ).sort_values("mean", ascending=False)

    def _extract_importance(self, model: BaseEstimator) -> Optional[np.ndarray]:
        """Helper to safely extract feature importance or coefficients."""
        # Unwrap pipeline if necessary
        if hasattr(model, "steps"):
            final_step = model.steps[-1][1]
        else:
            final_step = model

        if hasattr(final_step, "feature_importances_"):
            return final_step.feature_importances_
        elif hasattr(final_step, "coef_"):
            # For linear models, take absolute value of coefficients
            return np.abs(final_step.coef_.flatten())
        return None

    def check_production_readiness(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_odds_df: Optional[
            pd.DataFrame
        ] = None,  # Optional: Pass odds for ROI check
        min_auc: float = 0.55,
        max_ece: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Check if a model is ready for production.

        Best Practice:
        Includes Financial checks if odds data is provided.
        """
        metrics = self.evaluator.evaluate(
            model, X_test, y_test, model_name="production_candidate"
        )

        checks = {
            "auc_above_minimum": bool(metrics.auc >= min_auc),
            "calibration_acceptable": bool(metrics.calibration_error <= max_ece),
            "better_than_coin_flip": bool(metrics.auc > 0.5),
        }

        # Financial Check (if data available)
        if test_odds_df is not None:
            # Reconstruct dataframe for simulation
            df_sim = test_odds_df.copy()
            df_sim["home_probability"] = model.predict_proba(X_test)[:, 1]
            # Assumes y_test is 1 for Home Win, 0 for Away
            df_sim["outcome"] = y_test.map({1: "W", 0: "L"})

            sim_results = self.evaluator.calculate_profit_simulation(
                df_sim, home_odds_col="home_moneyline", away_odds_col="away_moneyline"
            )

            checks["is_profitable"] = bool(sim_results["total_profit"] > 0)
            metrics_dict = metrics.to_dict()
            metrics_dict["roi_simulation"] = sim_results["roi"]
        else:
            metrics_dict = metrics.to_dict()

        all_passed = all(checks.values())

        return {
            "production_ready": all_passed,
            "checks": checks,
            "metrics": metrics_dict,
        }

    def export_results(
        self,
        comparison_result: ComparisonResult,
        output_dir: Union[str, Path],
    ) -> None:
        """Export results to CSV and Text report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_result.rankings.to_csv(output_dir / "rankings.csv", index=False)

        # Text Report
        lines = ["MODEL BENCHMARK REPORT", "=" * 40, ""]
        lines.append(f"Best Model: {comparison_result.best_model_name}")
        lines.append(
            f"AUC Improvement: {comparison_result.improvement_over_baseline:.2f}%"
        )
        lines.append("\nTop 3 Models:")
        lines.append(comparison_result.rankings.head(3).to_string())

        (output_dir / "report.txt").write_text("\n".join(lines))
        print(f"Results exported to {output_dir}")
