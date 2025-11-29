"""
Synthetic data generation for NBA game predictions (V2).

Aligned with ml_game_features_v2 schema.
"""

import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["home_team", "away_team", "outcome"]
NUMERIC_COLS = [
    "home_moneyline",
    "away_moneyline",
    "home_team_rank",
    "home_days_rest",
    "home_team_avg_pts_scored",
    "home_team_avg_pts_scored_opp",
    "home_team_win_pct",
    "home_team_win_pct_last10",
    "home_star_score",
    "home_active_vorp",
    "home_pct_vorp_missing",
    "home_travel_miles_last_7_days",
    "home_games_last_7_days",
    "home_is_cross_country_trip",
    "away_team_rank",
    "away_days_rest",
    "away_team_avg_pts_scored",
    "away_team_avg_pts_scored_opp",
    "away_team_win_pct",
    "away_team_win_pct_last10",
    "away_star_score",
    "away_active_vorp",
    "away_pct_vorp_missing",
    "away_travel_miles_last_7_days",
    "away_games_last_7_days",
    "away_is_cross_country_trip",
    "travel_miles_differential",
    "star_score_differential",
    "active_vorp_differential",
]


class SyntheticDataGenerator:
    """
    Generate synthetic NBA game data aligned with ml_game_features_v2.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Mock team names for categorical generation
        self.teams = [f"Team_{i}" for i in range(30)]

    def generate_samples(
        self,
        n_samples: int = 1000,
        home_advantage: float = 0.06,  # ~6% bump for home teams
        noise_level: float = 0.35,
        start_date: str = "2023-10-24",
    ) -> pd.DataFrame:
        """
        Generate synthetic samples with correlated features.

        Best Practice: Uses Latent Variable generation.
        We generate a 'hidden' strength score for home/away, then derive
        stats from that strength so that Rank, Win Pct, and Pts Scored
        make sense together.
        """

        # 1. Generate Latent Team Strength (0.0 to 1.0)
        # Higher is better
        home_strength = self.rng.beta(2, 2, n_samples)
        away_strength = self.rng.beta(2, 2, n_samples)

        data = {}

        # --- Metadata ---
        data["game_date"] = self._generate_dates(start_date, n_samples)
        data["home_team"] = self.rng.choice(self.teams, n_samples)
        data["away_team"] = self.rng.choice(self.teams, n_samples)

        # --- Team Performance Metrics (Correlated with Strength) ---
        for prefix, strength in [("home", home_strength), ("away", away_strength)]:
            # Rank: Stronger teams have lower rank numbers (1 is best)
            # Add noise and clip to 1-30
            raw_rank = 30 - (strength * 29) + self.rng.normal(0, 2, n_samples)
            data[f"{prefix}_team_rank"] = np.clip(raw_rank, 1, 30).astype(int)

            # Win Pct: Directly correlates to strength
            data[f"{prefix}_team_win_pct"] = np.clip(
                strength + self.rng.normal(0, 0.05, n_samples), 0.1, 0.9
            ).round(3)

            # Recent Form (Last 10): Correlates to Win Pct but higher variance
            data[f"{prefix}_team_win_pct_last10"] = np.clip(
                data[f"{prefix}_team_win_pct"] + self.rng.normal(0, 0.15, n_samples),
                0.0,
                1.0,
            ).round(3)

            # Scoring
            avg_score = 112 + (strength * 10)  # 112 base + up to 10pts for strong teams
            data[f"{prefix}_team_avg_pts_scored"] = np.round(
                avg_score + self.rng.normal(0, 3, n_samples), 1
            )
            # Defense (Points Allowed): Strong teams allow fewer points
            avg_allowed = 118 - (strength * 10)
            data[f"{prefix}_team_avg_pts_scored_opp"] = np.round(
                avg_allowed + self.rng.normal(0, 3, n_samples), 1
            )

            # --- Roster / VORP Metrics ---
            # Star Score (0-5 scale of how many stars)
            data[f"{prefix}_star_score"] = np.clip(
                (strength * 4) + self.rng.normal(0, 1, n_samples), 0, 5
            ).astype(int)

            # Active VORP (Value Over Replacement Player)
            # Good teams have higher VORP
            data[f"{prefix}_active_vorp"] = np.round(
                (strength * 5) + self.rng.normal(0, 1, n_samples), 2
            )

            # Injuries (Pct VORP Missing) - Random, not strictly tied to strength
            data[f"{prefix}_pct_vorp_missing"] = np.clip(
                np.abs(self.rng.normal(0, 0.1, n_samples)), 0, 0.5
            ).round(3)

        # --- Schedule / Fatigue Metrics (Situational) ---
        for prefix in ["home", "away"]:
            # Rest: Poisson distribution centered around 1-2 days
            data[f"{prefix}_days_rest"] = np.clip(
                self.rng.poisson(1.5, n_samples), 0, 10
            ).astype(float)  # DDL says numeric

            # Travel
            data[f"{prefix}_games_last_7_days"] = self.rng.integers(1, 5, n_samples)
            data[f"{prefix}_travel_miles_last_7_days"] = np.clip(
                self.rng.exponential(1000, n_samples), 0, 5000
            ).round(1)

            # Cross Country (Binary 0/1)
            data[f"{prefix}_is_cross_country_trip"] = self.rng.binomial(
                1, 0.15, n_samples
            )

        # --- Calculated Differentials (Consistency Check) ---
        # Best Practice: Calculate these from the raw data, don't generate them independently
        # or the model will learn impossible relationships.
        data["travel_miles_differential"] = (
            data["home_travel_miles_last_7_days"]
            - data["away_travel_miles_last_7_days"]
        )
        data["star_score_differential"] = (
            data["home_star_score"] - data["away_star_score"]
        )
        data["active_vorp_differential"] = (
            data["home_active_vorp"] - data["away_active_vorp"]
        )

        # Create DataFrame
        df = pd.DataFrame(data)

        # --- Generate Outcome & Moneylines ---
        # We need the outcome logic first to determine the Moneylines (Reverse engineering)
        true_home_win_probs = self._calculate_win_probability(
            df, home_strength, away_strength, home_advantage, noise_level
        )

        # Determine actual outcome based on probability + noise
        outcomes_binary = (self.rng.random(n_samples) < true_home_win_probs).astype(int)
        df["outcome"] = np.where(outcomes_binary == 1, "W", "L")

        # Generate Moneylines based on the True Probability + Bookmaker "Vig"
        df["home_moneyline"], df["away_moneyline"] = self._generate_moneylines(
            true_home_win_probs
        )

        return df

    def _generate_dates(self, start_date_str: str, n: int) -> pd.Series:
        start = pd.to_datetime(start_date_str)
        # Add random number of days (0 to 180) to simulate a season
        offsets = pd.to_timedelta(self.rng.integers(0, 180, n), unit="D")

        # FIX: Wrap the DatetimeIndex in a Series to use the .dt accessor
        return pd.Series(start + offsets).dt.date

    def _calculate_win_probability(
        self,
        df: pd.DataFrame,
        home_strength: np.ndarray,
        away_strength: np.ndarray,
        home_advantage: float,
        noise_level: float = 0.35
    ) -> np.ndarray:
        """
        Calculate realistic win probability using logistic logic.
        Combines latent strength + specific situational columns.
        """
        # Base strength differential
        strength_diff = home_strength - away_strength

        # Situational modifiers
        # Fatigue: -0.05 prob for every game played > 3 in last week
        fatigue_home = np.where(df["home_games_last_7_days"] > 3, -0.05, 0)
        fatigue_away = np.where(df["away_games_last_7_days"] > 3, -0.05, 0)

        # Injuries
        injury_impact = (
            df["away_pct_vorp_missing"] - df["home_pct_vorp_missing"]
        ) * 0.2

        # Log-odds calculation
        log_odds = (
            (strength_diff * 4.0)
            + home_advantage
            + (fatigue_home - fatigue_away)
            + injury_impact
        )
        
        # FIX: Add Random Noise here!
        # This prevents the model from perfectly learning the formula.
        log_odds += self.rng.normal(0, noise_level, len(df))

        # Sigmoid function
        return 1 / (1 + np.exp(-log_odds))

    def _generate_moneylines(
        self, true_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic moneylines (US odds) based on win probabilities.
        Includes bookmaker margin (vig).
        """
        # Add 'Vig' (Bookmaker edge) - usually probability sums to ~1.05 instead of 1.0
        vig = 0.04
        implied_home = np.clip(true_probs + (vig / 2), 0.05, 0.95)
        implied_away = np.clip((1 - true_probs) + (vig / 2), 0.05, 0.95)

        def prob_to_moneyline(p):
            # US Odds conversion
            if p >= 0.5:
                return np.round(-(p / (1 - p)) * 100)
            else:
                return np.round(((1 - p) / p) * 100)

        # Vectorized conversion
        v_prob_to_ml = np.vectorize(prob_to_moneyline)

        return v_prob_to_ml(implied_home), v_prob_to_ml(implied_away)

    def augment_dataset(
        self,
        original_df: pd.DataFrame,
        augmentation_ratio: float = 0.5,
        perturbation_std: float = 0.05,
    ) -> pd.DataFrame:
        """
        Augment an existing dataset with slightly perturbed copies.
        Handles numeric columns only for perturbation.
        """
        n_synthetic = int(len(original_df) * augmentation_ratio)

        indices = self.rng.choice(len(original_df), size=n_synthetic, replace=True)
        synthetic = original_df.iloc[indices].copy().reset_index(drop=True)

        cols_to_perturb = [
            c
            for c in NUMERIC_COLS
            if c in synthetic.columns and "rank" not in c and "is_" not in c
        ]

        for col in cols_to_perturb:
            std = synthetic[col].std()
            if pd.isna(std) or std == 0:
                continue

            noise = self.rng.normal(0, perturbation_std * std, n_synthetic)
            synthetic[col] = synthetic[col] + noise

            # Sanity checks after perturbation
            if "pct" in col:
                synthetic[col] = np.clip(synthetic[col], 0, 1)
            if "score" in col or "pts" in col:
                synthetic[col] = np.maximum(synthetic[col], 0)

        return pd.concat([original_df, synthetic], ignore_index=True)
