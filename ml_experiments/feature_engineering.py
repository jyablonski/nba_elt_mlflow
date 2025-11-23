"""
Feature engineering utilities for NBA game predictions.

Provides feature transformations, derived features, and selection methods
to improve model performance.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
)
from sklearn.ensemble import RandomForestClassifier
from typing import Optional

from ml_experiments.config import FEATURE_COLUMNS, TARGET_COLUMN


class FeatureEngineer:
    """
    Feature engineering for NBA game prediction models.

    Provides methods for creating derived features, scaling,
    and feature selection.
    """

    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler: Optional[StandardScaler | MinMaxScaler | RobustScaler] = None
        self.selected_features: Optional[list[str]] = None
        self.feature_importances: Optional[pd.DataFrame] = None

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing columns.

        Adds differential features and interaction terms that
        may capture important patterns in the data.

        Args:
            df: Input DataFrame with base features

        Returns:
            DataFrame with additional derived features
        """
        result = df.copy()

        # Win percentage differential
        if "home_team_win_pct" in df.columns and "away_team_win_pct" in df.columns:
            result["win_pct_diff"] = df["home_team_win_pct"] - df["away_team_win_pct"]

        # Recent form differential
        if (
            "home_team_win_pct_last10" in df.columns
            and "away_team_win_pct_last10" in df.columns
        ):
            result["recent_form_diff"] = (
                df["home_team_win_pct_last10"] - df["away_team_win_pct_last10"]
            )

        # Rank differential (positive means home team is better ranked)
        if "home_team_rank" in df.columns and "away_team_rank" in df.columns:
            result["rank_diff"] = df["away_team_rank"] - df["home_team_rank"]

        # Net rating for each team (offensive - defensive efficiency proxy)
        if (
            "home_team_avg_pts_scored" in df.columns
            and "home_team_avg_pts_scored_opp" in df.columns
        ):
            result["home_net_rating"] = (
                df["home_team_avg_pts_scored"] - df["home_team_avg_pts_scored_opp"]
            )

        if (
            "away_team_avg_pts_scored" in df.columns
            and "away_team_avg_pts_scored_opp" in df.columns
        ):
            result["away_net_rating"] = (
                df["away_team_avg_pts_scored"] - df["away_team_avg_pts_scored_opp"]
            )

        # Net rating differential
        if "home_net_rating" in result.columns and "away_net_rating" in result.columns:
            result["net_rating_diff"] = (
                result["home_net_rating"] - result["away_net_rating"]
            )

        # Rest advantage
        if "home_days_rest" in df.columns and "away_days_rest" in df.columns:
            result["rest_diff"] = df["home_days_rest"] - df["away_days_rest"]
            # Binary flag for significant rest advantage (3+ days more rest)
            result["significant_rest_advantage"] = (result["rest_diff"] >= 3).astype(int)
            result["significant_rest_disadvantage"] = (result["rest_diff"] <= -3).astype(
                int
            )

        # Top players differential
        if "home_is_top_players" in df.columns and "away_is_top_players" in df.columns:
            result["top_players_diff"] = (
                df["home_is_top_players"] - df["away_is_top_players"]
            )

        # Momentum indicator (recent form vs season form)
        if (
            "home_team_win_pct" in df.columns
            and "home_team_win_pct_last10" in df.columns
        ):
            result["home_momentum"] = (
                df["home_team_win_pct_last10"] - df["home_team_win_pct"]
            )

        if (
            "away_team_win_pct" in df.columns
            and "away_team_win_pct_last10" in df.columns
        ):
            result["away_momentum"] = (
                df["away_team_win_pct_last10"] - df["away_team_win_pct"]
            )

        # Combined strength indicator
        if "win_pct_diff" in result.columns and "net_rating_diff" in result.columns:
            result["combined_strength"] = (
                result["win_pct_diff"] * 0.5 + result["net_rating_diff"] / 20 * 0.5
            )

        # Back-to-back game indicator
        if "home_days_rest" in df.columns:
            result["home_back_to_back"] = (df["home_days_rest"] == 0).astype(int)

        if "away_days_rest" in df.columns:
            result["away_back_to_back"] = (df["away_days_rest"] == 0).astype(int)

        return result

    def scale_features(
        self,
        df: pd.DataFrame,
        method: str = "standard",
        fit: bool = True,
        feature_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Scale numeric features using specified method.

        Args:
            df: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler (False for transform only)
            feature_columns: Columns to scale (default: all numeric)

        Returns:
            DataFrame with scaled features
        """
        result = df.copy()

        if feature_columns is None:
            feature_columns = [
                col
                for col in df.columns
                if col != TARGET_COLUMN and df[col].dtype in [np.float64, np.int64]
            ]

        if fit or self.scaler is None:
            if method == "standard":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            elif method == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            result[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        else:
            result[feature_columns] = self.scaler.transform(df[feature_columns])

        return result

    def select_features_statistical(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
        method: str = "f_classif",
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Select top k features using statistical tests.

        Args:
            X: Feature DataFrame
            y: Target series
            k: Number of features to select
            method: Statistical method ('f_classif', 'mutual_info')

        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        if method == "f_classif":
            selector = SelectKBest(f_classif, k=k)
        elif method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        selector.fit(X, y)
        mask = selector.get_support()
        self.selected_features = list(X.columns[mask])

        # Store feature scores
        scores = selector.scores_
        self.feature_importances = pd.DataFrame(
            {"feature": X.columns, "score": scores}
        ).sort_values("score", ascending=False)

        return X[self.selected_features], self.selected_features

    def select_features_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10,
        estimator: Optional[object] = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Select features using Recursive Feature Elimination.

        Args:
            X: Feature DataFrame
            y: Target series
            n_features: Number of features to select
            estimator: Estimator to use (default: RandomForest)

        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(X, y)

        mask = rfe.support_
        self.selected_features = list(X.columns[mask])

        # Store feature rankings
        self.feature_importances = pd.DataFrame(
            {"feature": X.columns, "ranking": rfe.ranking_}
        ).sort_values("ranking")

        return X[self.selected_features], self.selected_features

    def get_feature_importance(
        self,
        model,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """
        Extract feature importance from a trained model.

        Args:
            model: Trained model with feature_importances_ or coef_
            feature_names: List of feature names

        Returns:
            DataFrame with feature importances sorted by importance
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            raise ValueError("Model doesn't have feature_importances_ or coef_")

        self.feature_importances = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        return self.feature_importances

    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
        exclude_target: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Remove highly correlated features to reduce multicollinearity.

        Args:
            df: Input DataFrame
            threshold: Correlation threshold above which to remove features
            exclude_target: Whether to exclude target column from consideration

        Returns:
            Tuple of (DataFrame with removed features, list of removed features)
        """
        if exclude_target and TARGET_COLUMN in df.columns:
            feature_df = df.drop(columns=[TARGET_COLUMN])
        else:
            feature_df = df.copy()

        # Calculate correlation matrix
        corr_matrix = feature_df.corr().abs()

        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        result = df.drop(columns=to_drop)

        return result, to_drop

    def get_all_feature_names(self, include_derived: bool = True) -> list[str]:
        """
        Get list of all feature names (base and optionally derived).

        Args:
            include_derived: Whether to include derived feature names

        Returns:
            List of feature names
        """
        features = list(FEATURE_COLUMNS)

        if include_derived:
            derived = [
                "win_pct_diff",
                "recent_form_diff",
                "rank_diff",
                "home_net_rating",
                "away_net_rating",
                "net_rating_diff",
                "rest_diff",
                "significant_rest_advantage",
                "significant_rest_disadvantage",
                "top_players_diff",
                "home_momentum",
                "away_momentum",
                "combined_strength",
                "home_back_to_back",
                "away_back_to_back",
            ]
            features.extend(derived)

        return features
