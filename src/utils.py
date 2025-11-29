from datetime import datetime
import logging
from typing import Any, Dict

from joblib import load
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sqlalchemy.engine.base import Connection, Engine
from jyablonski_common_modules.sql import write_to_sql_upsert

from ml_experiments.training_pipeline import TrainingPipeline

# Set up module-level logger
logger = logging.getLogger(__name__)

# Constants
INPUT_ML_TABLE = "ml_game_features"
OUTPUT_ML_TABLE = "ml_game_predictions"

# Columns to drop for V1 ML prediction
ML_EXCLUDE_COLUMNS = [
    "home_team",
    "home_moneyline",
    "away_team",
    "away_moneyline",
    "game_date",
    "outcome",
]


def get_feature_flags(
    connection: Connection | Engine, schema: str = "gold"
) -> pd.DataFrame:
    """Retrieve feature flags from the database."""
    flags = pd.read_sql_query(
        sql=f"select * from {schema}.feature_flags", con=connection
    )
    logger.info(f"Retrieved {len(flags)} feature flags")
    return flags


def check_feature_flag(flag: str, flags_df: pd.DataFrame) -> bool:
    """Check if a specific feature flag is enabled."""
    flag_data = flags_df.query(f"flag == '{flag}'")

    if flag_data.empty:
        logger.info(f"Feature flag '{flag}' not found")
        return False

    is_enabled = bool(flag_data["is_enabled"].iloc[0])

    if not is_enabled:
        logger.info(f"Feature flag '{flag}' is disabled")

    return is_enabled


def pull_tonights_games(
    connection: Connection | Engine,
    schema: str,
    table: str = INPUT_ML_TABLE,
) -> pd.DataFrame:
    """Pull tonight's game features from the database."""
    games = pd.read_sql_query(sql=f"SELECT * FROM {schema}.{table}", con=connection)

    # Sort for consistency (handle missing column gracefully)
    if "home_team_avg_pts_scored" in games.columns:
        games = games.sort_values("home_team_avg_pts_scored")

    logger.info(f"Retrieved {len(games)} games from {schema}.{table}")
    return games


def load_ml_model(model_path: str) -> LogisticRegression:
    """Load a trained V1 ML model from disk (simple joblib)."""
    model = load(model_path)
    logger.info(f"Loaded V1 model from {model_path}")
    return model


def load_v2_artifacts(model_path: str) -> Dict[str, Any]:
    """
    Load V2 Production Artifacts.
    Returns dict with keys: 'model', 'feature_engineer', 'feature_names', etc.
    """
    try:
        artifacts = TrainingPipeline.load_artifacts(model_path)
        logger.info(f"Loaded V2 artifacts from {model_path}")
        return artifacts
    except FileNotFoundError:
        logger.error(f"V2 Model Artifact not found at {model_path}")
        return {}


def generate_win_predictions(
    games_df: pd.DataFrame,
    ml_model: LogisticRegression,
) -> pd.DataFrame:
    """Generate V1 win predictions."""
    if games_df.empty:
        logger.error("No data available for predictions")
        return pd.DataFrame()

    current_date = datetime.now().date()
    latest_date = pd.to_datetime(games_df["game_date"]).iloc[0].date()

    if latest_date != current_date:
        logger.error(
            f"Date mismatch: data is from {latest_date}, expected {current_date}"
        )
        return pd.DataFrame()

    # Prepare features for prediction
    ml_features = games_df.drop(columns=ML_EXCLUDE_COLUMNS)

    # Generate predictions
    predictions = ml_model.predict_proba(ml_features)
    prediction_df = pd.DataFrame(
        predictions,
        columns=["away_team_predicted_win_pct", "home_team_predicted_win_pct"],
    ).round(3)

    # Combine with original data
    result_df = games_df.reset_index(drop=True).drop(
        columns=["outcome"], errors="ignore"
    )
    result_df = pd.concat([result_df, prediction_df], axis=1)

    logger.info(f"Generated V1 predictions for {len(result_df)} games")
    return result_df


def generate_win_predictions_v2(
    games_df: pd.DataFrame,
    artifacts: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generate V2 win predictions using the FeatureEngineer pipeline.
    Returns the original DataFrame with two new prediction columns appended.
    """
    if games_df.empty:
        logger.warning("No V2 data available for predictions")
        return pd.DataFrame()

    # 1. Date Validation
    current_date = datetime.now().date()
    game_dates = pd.to_datetime(games_df["game_date"])
    latest_date = game_dates.iloc[0].date()

    if latest_date != current_date:
        logger.warning(
            f"V2 Date mismatch: data is from {latest_date}, expected {current_date}. Proceeding anyway."
        )

    # 2. Extract Artifacts
    model = artifacts["model"]
    engineer = artifacts["feature_engineer"]
    feature_names = artifacts["feature_names"]

    # 3. Preprocess Data (Inference Mode)
    try:
        # is_training=False ensures we use saved medians/scalers
        X_processed = engineer.preprocess_data(games_df, is_training=False)

        # Ensure exact column alignment with training data (Robustness)
        for col in feature_names:
            if col not in X_processed.columns:
                X_processed[col] = 0

        X_final = X_processed[feature_names]

    except Exception as e:
        logger.error(f"V2 Feature Engineering failed: {e}")
        return pd.DataFrame()

    # 4. Predict
    try:
        # model.predict_proba returns [Prob_Class_0, Prob_Class_1] (Loss, Win)
        # We grab index 1 for Home Win Probability
        home_probs = model.predict_proba(X_final)[:, 1]
    except Exception as e:
        logger.error(f"V2 Model Prediction failed: {e}")
        return pd.DataFrame()

    # 5. Format Output (Similar to V1 Process)
    # We clone the original DF so we retain all the feature columns (Rank, VORP, etc.)
    # needed for the destination DDL.
    result_df = games_df.reset_index(drop=True)

    # Drop 'outcome' if it exists in raw data (usually NULL in inference, but safe to drop)
    if "outcome" in result_df.columns:
        result_df = result_df.drop(columns=["outcome"])

    # Append the 2 new columns
    result_df["home_team_predicted_win_pct"] = np.round(home_probs, 3)
    result_df["away_team_predicted_win_pct"] = np.round(1.0 - home_probs, 3)

    logger.info(f"Generated V2 predictions for {len(result_df)} games")
    return result_df


def write_predictions_to_database(
    connection: Connection | Engine,
    predictions_df: pd.DataFrame,
    schema: str,
    table: str = OUTPUT_ML_TABLE,
    primary_keys: list[str] = None,
) -> None:
    """Write predictions to the database using upsert logic."""
    if primary_keys is None:
        primary_keys = ["home_team", "game_date"]

    if predictions_df.empty:
        logger.warning(f"Attempted to write empty dataframe to {table}")
        return

    write_to_sql_upsert(
        conn=connection,
        table=table,
        schema=schema,
        df=predictions_df,
        primary_keys=primary_keys,
    )

    logger.info(
        f"Successfully wrote {len(predictions_df)} predictions to {schema}.{table}"
    )

    return None
