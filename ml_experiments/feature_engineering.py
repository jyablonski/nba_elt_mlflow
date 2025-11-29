"""
Feature engineering utilities for NBA game predictions.

Provides feature transformations, derived features, and selection methods
aligned with the V2 Schema (VORP, Travel, Star Power).
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from ml_experiments.config import FEATURE_COLUMNS, TARGET_COLUMN


class FeatureEngineer:
    """
    Feature engineering pipeline for NBA game prediction.
    """

    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        self.imputer: Optional[SimpleImputer] = None
        self.selected_features: Optional[List[str]] = None
        self.feature_importances: Optional[pd.DataFrame] = None

        # NEW: Store the exact column order used during training
        self.numeric_cols_order: Optional[List[str]] = None

    def preprocess_data(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> pd.DataFrame:
        """
        Master method to run the full engineering pipeline.
        """
        df_eng = self.create_derived_features(df)

        if is_training:
            # --- TRAINING PHASE ---
            # 1. Identify Numeric Columns
            numeric_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()

            # Remove target from scaling/imputation
            if TARGET_COLUMN in numeric_cols:
                numeric_cols.remove(TARGET_COLUMN)

            # 2. SAVE THE ORDER (Critical Fix)
            self.numeric_cols_order = numeric_cols

            # 3. Clean Inf
            df_eng[numeric_cols] = df_eng[numeric_cols].replace(
                [np.inf, -np.inf], np.nan
            )

            # 4. Fit & Transform
            self.imputer = SimpleImputer(strategy="median")
            df_eng[numeric_cols] = self.imputer.fit_transform(df_eng[numeric_cols])

        else:
            # --- INFERENCE PHASE ---
            if self.numeric_cols_order is None:
                raise ValueError(
                    "FeatureEngineer has not been fitted. Call with is_training=True first."
                )

            # 1. Enforce Exact Column Order
            # Check for missing columns and fill with NaN (Imputer will handle them)
            missing_cols = [
                c for c in self.numeric_cols_order if c not in df_eng.columns
            ]
            if missing_cols:
                for c in missing_cols:
                    df_eng[c] = np.nan

            # Select columns in the STRICT order learned during training
            numeric_cols = self.numeric_cols_order

            # 2. Clean Inf
            df_eng[numeric_cols] = df_eng[numeric_cols].replace(
                [np.inf, -np.inf], np.nan
            )

            # 3. Transform (will no longer error on order mismatch)
            df_eng[numeric_cols] = self.imputer.transform(df_eng[numeric_cols])

        return df_eng

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features based on V2 Schema."""
        result = df.copy()

        # --- 1. Basic Differentials ---
        cols_to_diff = [
            ("win_pct", "home_team_win_pct", "away_team_win_pct"),
            ("recent_form", "home_team_win_pct_last10", "away_team_win_pct_last10"),
            ("rank", "away_team_rank", "home_team_rank"),
            ("star_score", "home_star_score", "away_star_score"),
            ("vorp", "home_active_vorp", "away_active_vorp"),
            (
                "travel",
                "home_travel_miles_last_7_days",
                "away_travel_miles_last_7_days",
            ),
        ]

        for name, home_col, away_col in cols_to_diff:
            if home_col in result.columns and away_col in result.columns:
                result[f"{name}_diff"] = result[home_col] - result[away_col]

        # --- 2. Advanced Metrics ---
        for side in ["home", "away"]:
            pts_scored = f"{side}_team_avg_pts_scored"
            pts_allowed = f"{side}_team_avg_pts_scored_opp"
            if pts_scored in result.columns and pts_allowed in result.columns:
                result[f"{side}_net_rating"] = result[pts_scored] - result[pts_allowed]

        if "home_net_rating" in result.columns and "away_net_rating" in result.columns:
            result["net_rating_diff"] = (
                result["home_net_rating"] - result["away_net_rating"]
            )

        # --- 3. Fatigue & Travel Impact ---
        result = self._calculate_fatigue_index(result)

        # --- 4. Momentum ---
        for side in ["home", "away"]:
            current = f"{side}_team_win_pct_last10"
            season = f"{side}_team_win_pct"
            if current in result.columns and season in result.columns:
                result[f"{side}_momentum"] = result[current] - result[season]

        return result

    def _calculate_fatigue_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate a composite fatigue score."""
        for side in ["home", "away"]:
            required = [
                f"{side}_days_rest",
                f"{side}_games_last_7_days",
                f"{side}_travel_miles_last_7_days",
            ]
            if not all(col in df.columns for col in required):
                continue

            # Weighted formula for fatigue
            # Fill NaNs with 0 temporarily for calculation if columns exist but have nulls
            rest = df[f"{side}_days_rest"].fillna(2)
            games = df[f"{side}_games_last_7_days"].fillna(2)
            miles = df[f"{side}_travel_miles_last_7_days"].fillna(500)

            rest_factor = (3 - rest.clip(upper=3)) * 1.5
            games_factor = games * 1.0
            travel_factor = np.log1p(miles) * 0.5

            cc_col = f"{side}_is_cross_country_trip"
            cc_factor = df[cc_col].fillna(0) * 2.0 if cc_col in df.columns else 0

            df[f"{side}_fatigue_index"] = (
                rest_factor + games_factor + travel_factor + cc_factor
            )

        if "home_fatigue_index" in df.columns and "away_fatigue_index" in df.columns:
            df["fatigue_diff"] = df["away_fatigue_index"] - df["home_fatigue_index"]

        return df

    # ... keep remaining methods (transform_skewed_features, select_features_*, etc.) ...
    def transform_skewed_features(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        result = df.copy()
        for col in columns:
            if col in result.columns:
                result[col] = np.log1p(result[col])
        return result

    def select_features_statistical(
        self, X: pd.DataFrame, y: pd.Series, k: int = 15, method: str = "mutual_info"
    ) -> Tuple[pd.DataFrame, List[str]]:
        X_numeric = X.select_dtypes(include=[np.number])
        if method == "f_classif":
            selector = SelectKBest(f_classif, k=k)
        elif method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        selector.fit(X_numeric, y)
        mask = selector.get_support()
        self.selected_features = list(X_numeric.columns[mask])
        scores = selector.scores_
        self.feature_importances = pd.DataFrame(
            {"feature": X_numeric.columns, "score": scores}
        ).sort_values("score", ascending=False)
        return X[self.selected_features], self.selected_features

    def select_features_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 15,
        estimator: Optional[object] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        X_numeric = X.select_dtypes(include=[np.number])
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(X_numeric, y)
        mask = rfe.support_
        self.selected_features = list(X_numeric.columns[mask])
        self.feature_importances = pd.DataFrame(
            {"feature": X_numeric.columns, "ranking": rfe.ranking_}
        ).sort_values("ranking")
        return X[self.selected_features], self.selected_features

    def remove_correlated_features(
        self, df: pd.DataFrame, threshold: float = 0.90, exclude_target: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        if exclude_target and TARGET_COLUMN in df.columns:
            feature_df = df.drop(columns=[TARGET_COLUMN])
        else:
            feature_df = df.copy()
        feature_df = feature_df.select_dtypes(include=[np.number])
        corr_matrix = feature_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        result = df.drop(columns=to_drop)
        return result, to_drop

    def get_feature_importance(self, model, feature_names: list[str]) -> pd.DataFrame:
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

    def get_all_feature_names(self, include_derived: bool = True) -> List[str]:
        features = list(FEATURE_COLUMNS)
        if include_derived:
            derived = [
                "win_pct_diff",
                "recent_form_diff",
                "rank_diff",
                "star_score_diff",
                "vorp_diff",
                "travel_diff",
                "home_net_rating",
                "away_net_rating",
                "net_rating_diff",
                "home_fatigue_index",
                "away_fatigue_index",
                "fatigue_diff",
                "home_momentum",
                "away_momentum",
            ]
            features.extend(derived)
        return features
