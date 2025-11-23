"""
Synthetic data generation for NBA game predictions.

Provides utilities to generate realistic synthetic training data
to augment existing datasets and improve model robustness.
"""

import numpy as np
import pandas as pd
from typing import Optional

from ml_experiments.config import FEATURE_COLUMNS, FEATURE_STATS, TARGET_COLUMN


class SyntheticDataGenerator:
    """
    Generate synthetic NBA game data for model training augmentation.

    Uses realistic feature distributions and correlations to create
    training samples that complement real historical data.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the synthetic data generator.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def generate_samples(
        self,
        n_samples: int = 1000,
        home_advantage: float = 0.06,
        noise_level: float = 0.1,
    ) -> pd.DataFrame:
        """
        Generate synthetic game samples with realistic feature distributions.

        Args:
            n_samples: Number of samples to generate
            home_advantage: Home court advantage factor (typical NBA ~6%)
            noise_level: Amount of randomness in outcome determination

        Returns:
            DataFrame with synthetic game data including features and outcomes
        """
        data = {}

        # Generate features for each column
        for feature in FEATURE_COLUMNS:
            data[feature] = self._generate_feature(feature, n_samples)

        df = pd.DataFrame(data)

        # Generate outcomes based on feature differences
        df[TARGET_COLUMN] = self._generate_outcomes(
            df, home_advantage=home_advantage, noise_level=noise_level
        )

        return df

    def _generate_feature(self, feature: str, n_samples: int) -> np.ndarray:
        """Generate values for a single feature based on its statistics."""
        stats = FEATURE_STATS[feature]

        if "values" in stats and "probs" in stats:
            # Categorical/ordinal feature with specific probabilities
            return self.rng.choice(stats["values"], size=n_samples, p=stats["probs"])
        else:
            # Continuous feature with normal distribution
            values = self.rng.normal(stats["mean"], stats["std"], n_samples)
            return np.clip(values, stats["min"], stats["max"])

    def _generate_outcomes(
        self,
        df: pd.DataFrame,
        home_advantage: float = 0.06,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Generate realistic game outcomes based on feature differences.

        Uses a logistic model considering:
        - Win percentage differential
        - Team ranking differential
        - Rest advantage
        - Top players availability
        - Home court advantage
        """
        n_samples = len(df)

        # Calculate team strength indicators
        win_pct_diff = df["home_team_win_pct"] - df["away_team_win_pct"]
        recent_form_diff = df["home_team_win_pct_last10"] - df["away_team_win_pct_last10"]

        # Rank difference (lower is better, so away - home)
        rank_diff = (df["away_team_rank"] - df["home_team_rank"]) / 30.0

        # Scoring differential (points scored - points allowed)
        home_net_rating = (
            df["home_team_avg_pts_scored"] - df["home_team_avg_pts_scored_opp"]
        )
        away_net_rating = (
            df["away_team_avg_pts_scored"] - df["away_team_avg_pts_scored_opp"]
        )
        net_rating_diff = (home_net_rating - away_net_rating) / 20.0

        # Rest advantage
        rest_diff = (df["home_days_rest"] - df["away_days_rest"]) / 7.0

        # Top players advantage
        players_diff = (df["home_is_top_players"] - df["away_is_top_players"]) / 2.0

        # Combined probability using logistic function
        log_odds = (
            0.3 * win_pct_diff
            + 0.2 * recent_form_diff
            + 0.15 * rank_diff
            + 0.2 * net_rating_diff
            + 0.05 * rest_diff
            + 0.1 * players_diff
            + home_advantage
        )

        # Add noise
        log_odds += self.rng.normal(0, noise_level, n_samples)

        # Convert to probability and sample outcomes
        probs = 1 / (1 + np.exp(-log_odds * 5))  # Scale factor for reasonable spread
        outcomes = (self.rng.random(n_samples) < probs).astype(int)

        return outcomes

    def augment_dataset(
        self,
        original_df: pd.DataFrame,
        augmentation_ratio: float = 0.5,
        perturbation_std: float = 0.05,
    ) -> pd.DataFrame:
        """
        Augment an existing dataset with perturbed copies and synthetic samples.

        Args:
            original_df: Original training dataset
            augmentation_ratio: Ratio of synthetic samples to add (0.5 = 50% more data)
            perturbation_std: Standard deviation for feature perturbation

        Returns:
            Combined DataFrame with original and augmented data
        """
        n_synthetic = int(len(original_df) * augmentation_ratio)

        # Generate perturbed copies of existing data
        perturbed = self._perturb_data(original_df, n_synthetic, perturbation_std)

        # Combine original and augmented data
        combined = pd.concat([original_df, perturbed], ignore_index=True)

        return combined

    def _perturb_data(
        self,
        df: pd.DataFrame,
        n_samples: int,
        perturbation_std: float,
    ) -> pd.DataFrame:
        """Create perturbed copies of existing samples."""
        # Sample indices with replacement
        indices = self.rng.choice(len(df), size=n_samples, replace=True)
        perturbed = df.iloc[indices].copy().reset_index(drop=True)

        # Add noise to continuous features
        for feature in FEATURE_COLUMNS:
            stats = FEATURE_STATS[feature]
            if "values" not in stats:  # Continuous feature
                noise = self.rng.normal(0, perturbation_std * stats["std"], n_samples)
                perturbed[feature] = np.clip(
                    perturbed[feature] + noise, stats["min"], stats["max"]
                )

        return perturbed

    def create_balanced_dataset(
        self,
        df: pd.DataFrame,
        balance_method: str = "oversample",
    ) -> pd.DataFrame:
        """
        Create a balanced dataset with equal class representation.

        Args:
            df: Input DataFrame with imbalanced classes
            balance_method: 'oversample' minority, 'undersample' majority, or 'smote'

        Returns:
            Balanced DataFrame
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame")

        class_counts = df[TARGET_COLUMN].value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()

        minority_df = df[df[TARGET_COLUMN] == minority_class]
        majority_df = df[df[TARGET_COLUMN] == majority_class]

        if balance_method == "oversample":
            # Oversample minority class
            n_oversample = len(majority_df) - len(minority_df)
            oversampled = self._perturb_data(minority_df, n_oversample, 0.02)
            oversampled[TARGET_COLUMN] = minority_class
            return pd.concat([df, oversampled], ignore_index=True)

        elif balance_method == "undersample":
            # Undersample majority class
            undersampled = majority_df.sample(
                n=len(minority_df), random_state=self.random_state
            )
            return pd.concat([undersampled, minority_df], ignore_index=True)

        else:
            raise ValueError(f"Unknown balance method: {balance_method}")

    def generate_edge_cases(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate edge case samples for robustness testing.

        Creates samples at the extremes of feature distributions
        to test model behavior in unusual situations.
        """
        data = {}

        for feature in FEATURE_COLUMNS:
            stats = FEATURE_STATS[feature]
            if "values" in stats:
                # Ordinal: alternate between extremes
                values = np.tile([stats["values"][0], stats["values"][-1]], n_samples // 2 + 1)[
                    :n_samples
                ]
            else:
                # Continuous: use min/max alternating
                values = np.tile([stats["min"], stats["max"]], n_samples // 2 + 1)[
                    :n_samples
                ]
            data[feature] = values

        df = pd.DataFrame(data)

        # Generate outcomes based on obvious winners/losers
        df[TARGET_COLUMN] = self._generate_outcomes(df, noise_level=0.05)

        return df

    def split_temporal(
        self,
        df: pd.DataFrame,
        date_column: str = "game_date",
        train_ratio: float = 0.8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally to simulate real-world prediction scenarios.

        This respects the time-series nature of sports data where
        you train on past games to predict future games.

        Args:
            df: DataFrame with game data
            date_column: Column containing game dates
            train_ratio: Proportion of data for training

        Returns:
            Tuple of (train_df, test_df)
        """
        if date_column in df.columns:
            df_sorted = df.sort_values(date_column)
        else:
            df_sorted = df.copy()

        split_idx = int(len(df_sorted) * train_ratio)
        train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
        test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)

        return train_df, test_df
