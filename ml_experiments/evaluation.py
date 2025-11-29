"""
Model evaluation utilities for NBA game predictions.

Provides comprehensive metrics, statistical tests, and vectorized
betting simulations for financial evaluation.
"""

from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict, List, Union

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
    matthews_corrcoef,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    mcc: float  # Matthews Correlation Coefficient
    log_loss: float
    brier_score: float
    confusion_matrix: List[List[int]]  # JSON serializable
    classification_report: str

    # Validation stats
    cv_accuracy_mean: Optional[float] = None
    cv_accuracy_std: Optional[float] = None
    calibration_error: Optional[float] = None  # ECE

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Generate a text summary of metrics."""
        lines = [
            f"Accuracy:     {self.accuracy:.4f}",
            f"Precision:    {self.precision:.4f}",
            f"Recall:       {self.recall:.4f}",
            f"F1 Score:     {self.f1:.4f}",
            f"MCC:          {self.mcc:.4f}",
            f"ROC AUC:      {self.auc:.4f}",
            f"Log Loss:     {self.log_loss:.4f}",
            f"Brier Score:  {self.brier_score:.4f}",
        ]
        if self.cv_accuracy_mean is not None:
            lines.append(
                f"CV Accuracy:  {self.cv_accuracy_mean:.4f} (+/- {self.cv_accuracy_std:.4f})"
            )
        if self.calibration_error is not None:
            lines.append(f"Calibration (ECE): {self.calibration_error:.4f}")
        return "\n".join(lines)


class ModelEvaluator:
    """
    Comprehensive model evaluation for NBA game predictions.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.results: Dict[str, EvaluationMetrics] = {}

    def evaluate(
        self,
        model,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        model_name: str = "model",
        cv_scores: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate a trained model on test data.
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
            mcc=matthews_corrcoef(y_test, y_pred),
            auc=roc_auc_score(y_test, y_proba),
            log_loss=log_loss(y_test, y_proba),
            brier_score=brier_score_loss(y_test, y_proba),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            classification_report=classification_report(y_test, y_pred),
        )

        # Add CV scores if provided
        if cv_scores is not None:
            metrics.cv_accuracy_mean = cv_scores.mean()
            metrics.cv_accuracy_std = cv_scores.std()

        # Calculate calibration error
        metrics.calibration_error = self._calculate_ece(y_test, y_proba)

        self.results[model_name] = metrics
        return metrics

    def compare_models(
        self,
        models: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: int = 5,
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset with Cross-Validation.
        """
        from sklearn.base import clone

        comparison_rows = []

        for name, model in models.items():
            print(f"Evaluating {name}...")

            # 1. Clone to ensure fresh start
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            # 2. CV scores (on training data to prevent leakage)
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="accuracy"
            )

            # 3. Evaluate on Hold-out Test Set
            metrics = self.evaluate(
                model_clone, X_test, y_test, model_name=name, cv_scores=cv_scores
            )

            # Flatten for DataFrame
            row = metrics.to_dict()
            row["model"] = name
            del row["confusion_matrix"]  # Remove non-scalar for dataframe
            del row["classification_report"]
            comparison_rows.append(row)

        df = pd.DataFrame(comparison_rows)
        return df.sort_values("auc", ascending=False)

    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        # Calculate weights for each bin
        bin_totals = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
        n_total = len(y_true)

        # Weighted absolute difference
        ece = (
            np.sum(bin_totals[bin_totals > 0] * np.abs(prob_true - prob_pred)) / n_total
        )
        return float(ece)

    # --- Financial / Betting Simulation ---

    @staticmethod
    def american_odds_to_prob(odds: np.ndarray) -> np.ndarray:
        """Vectorized conversion of American Odds to Implied Probability."""
        # Positive odds (e.g., +150): 100 / (odds + 100)
        # Negative odds (e.g., -110): -odds / (-odds + 100)

        # Avoid zero division
        safe_odds = np.where(odds == 0, 100, odds)

        pos_mask = safe_odds > 0
        neg_mask = ~pos_mask

        probs = np.zeros_like(safe_odds, dtype=float)

        probs[pos_mask] = 100 / (safe_odds[pos_mask] + 100)
        probs[neg_mask] = -safe_odds[neg_mask] / (-safe_odds[neg_mask] + 100)

        return probs

    def calculate_profit_simulation(
        self,
        df_results: pd.DataFrame,
        prob_col: str = "home_probability",
        target_col: str = "outcome",  # 'W' or 'L' for home team
        home_odds_col: str = "home_moneyline",
        away_odds_col: str = "away_moneyline",
        threshold: float = 0.5,
        kelly_fraction: float = 0.25,
        initial_bankroll: float = 1000.0,
    ) -> Dict[str, float]:
        """
        Simulate betting profit using vectorized operations (Best Practice).

        Args:
            df_results: DataFrame with probabilities, outcomes, and odds
            threshold: Minimum probability to place a bet
            kelly_fraction: Fraction of bankroll to bet (0.25 is conservative/standard)

        Returns:
            Dictionary with profit metrics
        """
        # Ensure we are working with arrays
        probs = df_results[prob_col].values
        # Convert target to binary (1 if Home Won, 0 if Away Won)
        y_true = (df_results[target_col].astype(str) == "W").astype(int).values

        home_odds = df_results[home_odds_col].values
        away_odds = df_results[away_odds_col].values

        # 1. Calculate Implied Probabilities
        implied_home = self.american_odds_to_prob(home_odds)
        implied_away = self.american_odds_to_prob(away_odds)

        # 2. Identify Edges
        # Edge = Model Prob - Implied Prob
        edge_home = probs - implied_home
        edge_away = (1 - probs) - implied_away

        # 3. Determine Bets (Vectorized)
        # Bet Home if edge > 0 AND prob > threshold
        bet_home_mask = (edge_home > 0) & (probs >= threshold)

        # Bet Away if edge > 0 AND (1-prob) >= threshold
        bet_away_mask = (edge_away > 0) & ((1 - probs) >= threshold)

        # 4. Calculate decimal odds for payout calc
        # Decimal = 1 / Implied (approx, ignoring vig for payout math)
        # Better: Convert American to Decimal directly
        def get_decimal(us_odds):
            return np.where(us_odds > 0, 1 + (us_odds / 100), 1 + (100 / -us_odds))

        dec_odds_home = get_decimal(home_odds)
        dec_odds_away = get_decimal(away_odds)

        # 5. Calculate Kelly Stake % (size of bankroll)
        # Kelly = Edge / (Decimal Odds - 1)
        stake_pct_home = np.where(
            bet_home_mask, (edge_home / (dec_odds_home - 1)) * kelly_fraction, 0
        )
        stake_pct_away = np.where(
            bet_away_mask, (edge_away / (dec_odds_away - 1)) * kelly_fraction, 0
        )

        # Clip stakes to max 5% of bankroll for safety
        stake_pct_home = np.clip(stake_pct_home, 0, 0.05)
        stake_pct_away = np.clip(stake_pct_away, 0, 0.05)

        # 6. Simulate Bankroll Process
        # Note: Vectorization of compounding bankroll is hard because state depends on previous.
        # However, for metric calculation, we can assume fixed betting units or
        # use a loop for the final cumsum. Given N=1000s, a loop is fine here for the compounding.

        bankroll = initial_bankroll
        history = []
        n_bets = 0
        n_wins = 0

        for i in range(len(probs)):
            current_stake = 0
            pnl = 0

            if stake_pct_home[i] > 0:
                n_bets += 1
                current_stake = bankroll * stake_pct_home[i]
                if y_true[i] == 1:  # Home Won
                    profit = current_stake * (dec_odds_home[i] - 1)
                    bankroll += profit
                    pnl = profit
                    n_wins += 1
                else:
                    bankroll -= current_stake
                    pnl = -current_stake

            elif stake_pct_away[i] > 0:
                n_bets += 1
                current_stake = bankroll * stake_pct_away[i]
                if y_true[i] == 0:  # Away Won (Home Lost)
                    profit = current_stake * (dec_odds_away[i] - 1)
                    bankroll += profit
                    pnl = profit
                    n_wins += 1
                else:
                    bankroll -= current_stake
                    pnl = -current_stake

            history.append(bankroll)

        return {
            "final_bankroll": bankroll,
            "total_profit": bankroll - initial_bankroll,
            "roi": ((bankroll - initial_bankroll) / initial_bankroll) * 100,
            "bets_placed": n_bets,
            "win_rate": n_wins / n_bets if n_bets > 0 else 0.0,
        }
