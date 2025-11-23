"""
Model comparison and benchmarking utilities.

Provides comprehensive tools for comparing models, running benchmarks,
and selecting the best model for production deployment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    learning_curve,
    validation_curve,
)
from scipy import stats

from ml_experiments.models import ModelFactory, create_baseline_model
from ml_experiments.evaluation import (
    ModelEvaluator,
    EvaluationMetrics,
    calculate_baseline_accuracy,
)
from ml_experiments.training_pipeline import TrainingResult


@dataclass
class ComparisonResult:
    """Results from model comparison."""

    rankings: pd.DataFrame
    detailed_metrics: dict[str, EvaluationMetrics]
    statistical_tests: pd.DataFrame
    best_model_name: str
    best_model: BaseEstimator
    improvement_over_baseline: float


class ModelComparison:
    """
    Comprehensive model comparison and benchmarking.

    Provides methods for comparing models across multiple metrics,
    statistical significance testing, and production readiness assessment.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the model comparison framework.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.factory = ModelFactory()
        self.evaluator = ModelEvaluator()
        self.comparison_results: dict[str, dict[str, Any]] = {}

    def compare_models(
        self,
        models: dict[str, BaseEstimator],
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        cv_folds: int = 5,
        include_baseline: bool = True,
    ) -> ComparisonResult:
        """
        Compare multiple models comprehensively.

        Args:
            models: Dictionary of model_name -> model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            cv_folds: Number of cross-validation folds
            include_baseline: Whether to include baseline model

        Returns:
            ComparisonResult with rankings and analysis
        """
        all_models = dict(models)

        # Add baseline if requested
        if include_baseline and "baseline" not in all_models:
            all_models["baseline"] = create_baseline_model()

        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_state
        )

        results = []
        cv_scores_dict = {}
        detailed_metrics = {}

        for name, model in all_models.items():
            print(f"Evaluating {name}...")

            # Clone and fit
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            # Cross-validation scores
            cv_accuracy = cross_val_score(model_clone, X_train, y_train, cv=cv, scoring="accuracy")
            cv_roc_auc = cross_val_score(model_clone, X_train, y_train, cv=cv, scoring="roc_auc")
            cv_f1 = cross_val_score(model_clone, X_train, y_train, cv=cv, scoring="f1")
            cv_neg_log_loss = cross_val_score(
                model_clone, X_train, y_train, cv=cv, scoring="neg_log_loss"
            )

            cv_scores_dict[name] = {
                "accuracy": cv_accuracy,
                "roc_auc": cv_roc_auc,
                "f1": cv_f1,
                "log_loss": -cv_neg_log_loss,
            }

            # Test set evaluation
            metrics = self.evaluator.evaluate(
                model_clone, X_test, y_test, model_name=name, cv_scores=cv_accuracy
            )
            detailed_metrics[name] = metrics

            self.comparison_results[name] = {
                "model": model_clone,
                "cv_scores": cv_scores_dict[name],
                "metrics": metrics,
            }

            results.append({
                "model": name,
                "cv_accuracy_mean": cv_accuracy.mean(),
                "cv_accuracy_std": cv_accuracy.std(),
                "cv_roc_auc_mean": cv_roc_auc.mean(),
                "cv_roc_auc_std": cv_roc_auc.std(),
                "cv_f1_mean": cv_f1.mean(),
                "cv_f1_std": cv_f1.std(),
                "cv_log_loss_mean": (-cv_neg_log_loss).mean(),
                "test_accuracy": metrics.accuracy,
                "test_roc_auc": metrics.roc_auc,
                "test_brier": metrics.brier_score,
                "calibration_error": metrics.calibration_error,
            })

        # Create rankings DataFrame
        rankings = pd.DataFrame(results)
        rankings = rankings.sort_values("test_roc_auc", ascending=False).reset_index(drop=True)
        rankings["rank"] = rankings.index + 1

        # Statistical significance tests
        stat_tests = self._run_statistical_tests(cv_scores_dict)

        # Find best model
        best_name = rankings.iloc[0]["model"]
        best_model = self.comparison_results[best_name]["model"]

        # Calculate improvement over baseline
        if include_baseline:
            baseline_auc = rankings[rankings["model"] == "baseline"]["test_roc_auc"].values[0]
            best_auc = rankings.iloc[0]["test_roc_auc"]
            improvement = (best_auc - baseline_auc) / baseline_auc * 100
        else:
            improvement = 0.0

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
        cv_scores_dict: dict[str, dict[str, np.ndarray]],
    ) -> pd.DataFrame:
        """Run pairwise statistical tests between models."""
        model_names = list(cv_scores_dict.keys())
        test_results = []

        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1:]:
                scores_a = cv_scores_dict[model_a]["roc_auc"]
                scores_b = cv_scores_dict[model_b]["roc_auc"]

                # Paired t-test
                t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

                test_results.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_diff": scores_a.mean() - scores_b.mean(),
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant_05": p_value < 0.05,
                    "significant_01": p_value < 0.01,
                })

        return pd.DataFrame(test_results)

    def generate_learning_curves(
        self,
        model: BaseEstimator,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        train_sizes: Optional[np.ndarray] = None,
        cv: int = 5,
    ) -> dict[str, np.ndarray]:
        """
        Generate learning curves to analyze model behavior.

        Args:
            model: Model to analyze
            X: Features
            y: Labels
            train_sizes: Training set sizes to evaluate
            cv: Number of CV folds

        Returns:
            Dictionary with training sizes and scores
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes_abs, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            train_sizes=train_sizes,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            random_state=self.random_state,
        )

        return {
            "train_sizes": train_sizes_abs,
            "train_scores_mean": train_scores.mean(axis=1),
            "train_scores_std": train_scores.std(axis=1),
            "test_scores_mean": test_scores.mean(axis=1),
            "test_scores_std": test_scores.std(axis=1),
        }

    def analyze_feature_importance_stability(
        self,
        model: BaseEstimator,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        feature_names: list[str],
        n_iterations: int = 10,
    ) -> pd.DataFrame:
        """
        Analyze stability of feature importances across multiple runs.

        Args:
            model: Model with feature_importances_ attribute
            X: Features
            y: Labels
            feature_names: List of feature names
            n_iterations: Number of bootstrap iterations

        Returns:
            DataFrame with importance statistics
        """
        importance_matrix = []

        for i in range(n_iterations):
            # Bootstrap sample
            rng = np.random.default_rng(self.random_state + i)
            indices = rng.choice(len(X), size=len(X), replace=True)

            if isinstance(X, pd.DataFrame):
                X_boot = X.iloc[indices]
            else:
                X_boot = X[indices]

            if isinstance(y, pd.Series):
                y_boot = y.iloc[indices]
            else:
                y_boot = y[indices]

            # Train model
            model_clone = clone(model)
            model_clone.fit(X_boot, y_boot)

            if hasattr(model_clone, "feature_importances_"):
                importance_matrix.append(model_clone.feature_importances_)

        if not importance_matrix:
            raise ValueError("Model doesn't have feature_importances_")

        importance_array = np.array(importance_matrix)

        return pd.DataFrame({
            "feature": feature_names,
            "importance_mean": importance_array.mean(axis=0),
            "importance_std": importance_array.std(axis=0),
            "importance_min": importance_array.min(axis=0),
            "importance_max": importance_array.max(axis=0),
            "cv_coefficient": (
                importance_array.std(axis=0) / importance_array.mean(axis=0)
            ),
        }).sort_values("importance_mean", ascending=False)

    def check_production_readiness(
        self,
        model: BaseEstimator,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        baseline_accuracy: Optional[float] = None,
        min_accuracy: float = 0.55,
        min_auc: float = 0.55,
        max_calibration_error: float = 0.1,
    ) -> dict[str, Any]:
        """
        Check if a model is ready for production deployment.

        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            baseline_accuracy: Baseline to compare against
            min_accuracy: Minimum required accuracy
            min_auc: Minimum required AUC
            max_calibration_error: Maximum acceptable calibration error

        Returns:
            Dictionary with readiness assessment
        """
        metrics = self.evaluator.evaluate(
            model, X_test, y_test, model_name="production_candidate"
        )

        if baseline_accuracy is None:
            baseline_accuracy = calculate_baseline_accuracy(y_test)

        checks = {
            "accuracy_above_baseline": metrics.accuracy > baseline_accuracy,
            "accuracy_above_minimum": metrics.accuracy >= min_accuracy,
            "auc_above_minimum": metrics.roc_auc >= min_auc,
            "calibration_acceptable": (
                metrics.calibration_error is not None
                and metrics.calibration_error <= max_calibration_error
            ),
            "better_than_random": metrics.roc_auc > 0.5,
        }

        all_passed = all(checks.values())

        return {
            "production_ready": all_passed,
            "checks": checks,
            "metrics": metrics.to_dict(),
            "recommendations": self._generate_recommendations(checks, metrics),
        }

    def _generate_recommendations(
        self,
        checks: dict[str, bool],
        metrics: EvaluationMetrics,
    ) -> list[str]:
        """Generate recommendations based on check results."""
        recommendations = []

        if not checks["accuracy_above_baseline"]:
            recommendations.append(
                "Model performs worse than always predicting majority class. "
                "Consider feature engineering or different model architecture."
            )

        if not checks["accuracy_above_minimum"]:
            recommendations.append(
                f"Accuracy ({metrics.accuracy:.3f}) is below minimum threshold. "
                "Try hyperparameter tuning or ensemble methods."
            )

        if not checks["auc_above_minimum"]:
            recommendations.append(
                f"ROC AUC ({metrics.roc_auc:.3f}) is too low. "
                "Model may not discriminate well between classes."
            )

        if not checks["calibration_acceptable"]:
            recommendations.append(
                "Probability calibration is poor. "
                "Consider using CalibratedClassifierCV for better probabilities."
            )

        if not recommendations:
            recommendations.append("Model appears ready for production deployment.")

        return recommendations

    def generate_comparison_report(
        self,
        comparison_result: ComparisonResult,
    ) -> str:
        """
        Generate a text report from comparison results.

        Args:
            comparison_result: Results from compare_models

        Returns:
            Formatted text report
        """
        lines = [
            "=" * 60,
            "MODEL COMPARISON REPORT",
            "=" * 60,
            "",
            "RANKINGS (by Test ROC AUC)",
            "-" * 40,
        ]

        for _, row in comparison_result.rankings.iterrows():
            lines.append(
                f"{row['rank']:2d}. {row['model']:25s} "
                f"AUC: {row['test_roc_auc']:.4f} "
                f"Acc: {row['test_accuracy']:.4f}"
            )

        lines.extend([
            "",
            f"Best Model: {comparison_result.best_model_name}",
            f"Improvement over baseline: {comparison_result.improvement_over_baseline:+.2f}%",
            "",
            "CROSS-VALIDATION SUMMARY",
            "-" * 40,
        ])

        for _, row in comparison_result.rankings.iterrows():
            lines.append(
                f"{row['model']:25s} "
                f"CV Acc: {row['cv_accuracy_mean']:.4f} (+/- {row['cv_accuracy_std']:.4f})"
            )

        # Statistical significance
        sig_tests = comparison_result.statistical_tests
        if len(sig_tests) > 0:
            sig_pairs = sig_tests[sig_tests["significant_05"]]
            if len(sig_pairs) > 0:
                lines.extend([
                    "",
                    "SIGNIFICANT DIFFERENCES (p < 0.05)",
                    "-" * 40,
                ])
                for _, row in sig_pairs.iterrows():
                    winner = row["model_a"] if row["mean_diff"] > 0 else row["model_b"]
                    lines.append(
                        f"{row['model_a']} vs {row['model_b']}: "
                        f"p={row['p_value']:.4f} ({winner} better)"
                    )

        lines.extend(["", "=" * 60])

        return "\n".join(lines)

    def export_results(
        self,
        comparison_result: ComparisonResult,
        output_dir: str | Path,
    ) -> None:
        """
        Export comparison results to files.

        Args:
            comparison_result: Results to export
            output_dir: Directory to save files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save rankings
        comparison_result.rankings.to_csv(
            output_dir / "model_rankings.csv", index=False
        )

        # Save statistical tests
        comparison_result.statistical_tests.to_csv(
            output_dir / "statistical_tests.csv", index=False
        )

        # Save report
        report = self.generate_comparison_report(comparison_result)
        (output_dir / "comparison_report.txt").write_text(report)

        print(f"Results exported to {output_dir}")
