"""
Model evaluation utilities for NBA game predictions.

Provides comprehensive metrics and evaluation methods for
comparing model performance.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, cross_val_predict


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss_value: float
    brier_score: float
    confusion_mat: np.ndarray
    classification_rep: str

    # Optional additional metrics
    cv_accuracy_mean: Optional[float] = None
    cv_accuracy_std: Optional[float] = None
    calibration_error: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "log_loss": self.log_loss_value,
            "brier_score": self.brier_score,
            "cv_accuracy_mean": self.cv_accuracy_mean,
            "cv_accuracy_std": self.cv_accuracy_std,
            "calibration_error": self.calibration_error,
        }

    def summary(self) -> str:
        """Generate a text summary of metrics."""
        lines = [
            f"Accuracy:     {self.accuracy:.4f}",
            f"Precision:    {self.precision:.4f}",
            f"Recall:       {self.recall:.4f}",
            f"F1 Score:     {self.f1:.4f}",
            f"ROC AUC:      {self.roc_auc:.4f}",
            f"Log Loss:     {self.log_loss_value:.4f}",
            f"Brier Score:  {self.brier_score:.4f}",
        ]
        if self.cv_accuracy_mean is not None:
            lines.append(
                f"CV Accuracy:  {self.cv_accuracy_mean:.4f} (+/- {self.cv_accuracy_std:.4f})"
            )
        if self.calibration_error is not None:
            lines.append(f"Calibration:  {self.calibration_error:.4f}")
        return "\n".join(lines)


class ModelEvaluator:
    """
    Comprehensive model evaluation for NBA game predictions.

    Provides methods for calculating various performance metrics,
    probability calibration analysis, and statistical tests.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.results: dict[str, EvaluationMetrics] = {}

    def evaluate(
        self,
        model,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        model_name: str = "model",
        cv_scores: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model with predict and predict_proba methods
            X_test: Test features
            y_test: Test labels
            model_name: Name for storing results
            cv_scores: Optional cross-validation scores

        Returns:
            EvaluationMetrics object with all computed metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)

        # Get probability predictions if available
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        # Calculate metrics
        metrics = EvaluationMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_test, y_proba),
            log_loss_value=log_loss(y_test, y_proba),
            brier_score=brier_score_loss(y_test, y_proba),
            confusion_mat=confusion_matrix(y_test, y_pred),
            classification_rep=classification_report(y_test, y_pred),
        )

        # Add CV scores if provided
        if cv_scores is not None:
            metrics.cv_accuracy_mean = cv_scores.mean()
            metrics.cv_accuracy_std = cv_scores.std()

        # Calculate calibration error
        metrics.calibration_error = self._calculate_calibration_error(y_test, y_proba)

        self.results[model_name] = metrics
        return metrics

    def evaluate_with_cv(
        self,
        model,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        model_name: str = "model",
        cv: int = 5,
    ) -> EvaluationMetrics:
        """
        Evaluate a model using cross-validation.

        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            model_name: Name for storing results
            cv: Number of cross-validation folds

        Returns:
            EvaluationMetrics with cross-validated scores
        """
        from sklearn.base import clone

        # Cross-validation scores
        cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_roc_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")

        # Get cross-validated predictions
        y_pred_cv = cross_val_predict(model, X, y, cv=cv)
        y_proba_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

        metrics = EvaluationMetrics(
            accuracy=cv_accuracy.mean(),
            precision=precision_score(y, y_pred_cv, zero_division=0),
            recall=recall_score(y, y_pred_cv, zero_division=0),
            f1=cv_f1.mean(),
            roc_auc=cv_roc_auc.mean(),
            log_loss_value=log_loss(y, y_proba_cv),
            brier_score=brier_score_loss(y, y_proba_cv),
            confusion_mat=confusion_matrix(y, y_pred_cv),
            classification_rep=classification_report(y, y_pred_cv),
            cv_accuracy_mean=cv_accuracy.mean(),
            cv_accuracy_std=cv_accuracy.std(),
        )

        metrics.calibration_error = self._calculate_calibration_error(y, y_proba_cv)
        self.results[model_name] = metrics

        return metrics

    def _calculate_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Lower is better - indicates how well predicted probabilities
        match actual outcomes.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        bin_totals = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
        non_empty_bins = bin_totals > 0

        if non_empty_bins.sum() == 0:
            return 0.0

        # ECE: weighted average of |accuracy - confidence|
        ece = np.sum(
            bin_totals[: len(prob_true)]
            * np.abs(prob_true - prob_pred)
            / bin_totals[: len(prob_true)].sum()
        )

        return float(ece)

    def compare_models(
        self,
        models: dict[str, Any],
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        cv: int = 5,
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.

        Args:
            models: Dictionary of model_name -> model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            cv: Number of CV folds

        Returns:
            DataFrame with comparison metrics for all models
        """
        from sklearn.base import clone

        results = []

        for name, model in models.items():
            print(f"Evaluating {name}...")

            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            # Cross-validation scores on training data
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv)

            # Evaluate on test data
            metrics = self.evaluate(
                model_clone, X_test, y_test, model_name=name, cv_scores=cv_scores
            )

            results.append({"model": name, **metrics.to_dict()})

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values("roc_auc", ascending=False)

        return comparison_df

    def calculate_profit_simulation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        odds_home: np.ndarray,
        odds_away: np.ndarray,
        threshold: float = 0.5,
        kelly_fraction: float = 0.25,
    ) -> dict[str, float]:
        """
        Simulate betting profit using model predictions.

        This helps evaluate the practical value of the model
        for sports betting applications.

        Args:
            y_true: Actual outcomes (1 = home win)
            y_pred: Predicted outcomes
            y_proba: Predicted probabilities for home win
            odds_home: Home team moneyline odds
            odds_away: Away team moneyline odds
            threshold: Probability threshold for betting
            kelly_fraction: Fraction of Kelly criterion to use

        Returns:
            Dictionary with profit metrics
        """
        n_games = len(y_true)
        starting_bankroll = 1000.0
        bankroll = starting_bankroll
        n_bets = 0
        n_wins = 0

        for i in range(n_games):
            prob_home = y_proba[i]
            prob_away = 1 - prob_home

            # Convert American odds to implied probability
            if odds_home[i] > 0:
                implied_home = 100 / (odds_home[i] + 100)
            else:
                implied_home = -odds_home[i] / (-odds_home[i] + 100)

            if odds_away[i] > 0:
                implied_away = 100 / (odds_away[i] + 100)
            else:
                implied_away = -odds_away[i] / (-odds_away[i] + 100)

            # Check for positive expected value opportunities
            edge_home = prob_home - implied_home
            edge_away = prob_away - implied_away

            bet_amount = 0
            bet_on_home = None

            if edge_home > 0.05 and prob_home >= threshold:
                # Bet on home team
                kelly = edge_home / (1 - implied_home)
                bet_amount = bankroll * kelly * kelly_fraction
                bet_on_home = True
            elif edge_away > 0.05 and prob_away >= threshold:
                # Bet on away team
                kelly = edge_away / (1 - implied_away)
                bet_amount = bankroll * kelly * kelly_fraction
                bet_on_home = False

            if bet_amount > 0:
                n_bets += 1
                actual_home_win = y_true[i] == 1

                if bet_on_home == actual_home_win:
                    n_wins += 1
                    if bet_on_home:
                        if odds_home[i] > 0:
                            profit = bet_amount * odds_home[i] / 100
                        else:
                            profit = bet_amount * 100 / -odds_home[i]
                    else:
                        if odds_away[i] > 0:
                            profit = bet_amount * odds_away[i] / 100
                        else:
                            profit = bet_amount * 100 / -odds_away[i]
                    bankroll += profit
                else:
                    bankroll -= bet_amount

        return {
            "final_bankroll": bankroll,
            "profit": bankroll - starting_bankroll,
            "roi": (bankroll - starting_bankroll) / starting_bankroll * 100,
            "n_bets": n_bets,
            "n_wins": n_wins,
            "win_rate": n_wins / n_bets if n_bets > 0 else 0,
        }

    def get_results_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of all evaluated models."""
        if not self.results:
            return pd.DataFrame()

        rows = [
            {"model": name, **metrics.to_dict()}
            for name, metrics in self.results.items()
        ]
        return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

    def statistical_significance_test(
        self,
        model_a_scores: np.ndarray,
        model_b_scores: np.ndarray,
        test: str = "paired_ttest",
    ) -> dict[str, float]:
        """
        Test statistical significance between two models' CV scores.

        Args:
            model_a_scores: CV scores for model A
            model_b_scores: CV scores for model B
            test: Statistical test to use ('paired_ttest', 'wilcoxon')

        Returns:
            Dictionary with test statistic and p-value
        """
        from scipy import stats

        if test == "paired_ttest":
            statistic, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
        elif test == "wilcoxon":
            statistic, p_value = stats.wilcoxon(model_a_scores, model_b_scores)
        else:
            raise ValueError(f"Unknown test: {test}")

        return {
            "statistic": statistic,
            "p_value": p_value,
            "significant_at_05": p_value < 0.05,
            "significant_at_01": p_value < 0.01,
        }


def calculate_baseline_accuracy(y: np.ndarray | pd.Series) -> float:
    """
    Calculate baseline accuracy (predicting majority class).

    Args:
        y: Target labels

    Returns:
        Baseline accuracy
    """
    y_arr = np.array(y)
    majority_class = np.bincount(y_arr).argmax()
    return (y_arr == majority_class).mean()


def calculate_random_baseline(n_samples: int, class_ratio: float = 0.5) -> float:
    """
    Calculate expected accuracy of random guessing.

    Args:
        n_samples: Number of samples
        class_ratio: Ratio of positive class

    Returns:
        Expected random accuracy
    """
    return class_ratio**2 + (1 - class_ratio) ** 2
