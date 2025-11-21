from datetime import datetime
import logging

from joblib import load
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy.engine.base import Connection, Engine
from jyablonski_common_modules.sql import write_to_sql_upsert

# Set up module-level logger
logger = logging.getLogger(__name__)

# Constants
INPUT_ML_TABLE = "ml_game_features"
OUTPUT_ML_TABLE = "ml_game_predictions"

# Columns to drop for ML prediction
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
    """
    Retrieve feature flags from the database.

    Args:
        connection: Database connection
        schema: Schema where feature_flags table is located

    Returns:
        DataFrame containing feature flags
    """
    flags = pd.read_sql_query(
        sql=f"select * from {schema}.feature_flags", con=connection
    )
    logger.info(f"Retrieved {len(flags)} feature flags")
    return flags


def check_feature_flag(flag: str, flags_df: pd.DataFrame) -> bool:
    """
    Check if a specific feature flag is enabled.

    Args:
        flag: Name of the feature flag to check
        flags_df: DataFrame containing feature flags

    Returns:
        True if flag exists and is enabled, False otherwise
    """
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
    """
    Pull tonight's game features from the database.

    Args:
        connection: Database connection
        schema: Schema where the games table is located
        table: Table name containing game features

    Returns:
        DataFrame with tonight's games, sorted by home_team_avg_pts_scored
    """
    games = pd.read_sql_query(
        sql=f"SELECT * FROM {schema}.{table}", con=connection
    ).sort_values("home_team_avg_pts_scored")

    logger.info(f"Retrieved {len(games)} games from {schema}.{table}")
    return games


def load_ml_model(model_path: str) -> LogisticRegression:
    """
    Load a trained ML model from disk.

    Args:
        model_path: Path to the joblib model file

    Returns:
        Loaded ML model
    """
    model = load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model


def generate_win_predictions(
    games_df: pd.DataFrame,
    ml_model: LogisticRegression,
) -> pd.DataFrame:
    """
    Generate win prediction percentages for games.

    Args:
        games_df: DataFrame with game features and metadata
        ml_model: Trained logistic regression model

    Returns:
        DataFrame with predictions, or empty DataFrame if validation fails
    """
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
    result_df = games_df.reset_index(drop=True).drop(columns=["outcome"])
    result_df = pd.concat([result_df, prediction_df], axis=1)

    logger.info(f"Generated predictions for {len(result_df)} games")
    return result_df


def write_predictions_to_database(
    connection: Connection | Engine,
    predictions_df: pd.DataFrame,
    schema: str,
    table: str = OUTPUT_ML_TABLE,
    primary_keys: list[str] = None,
) -> None:
    """
    Write predictions to the database using upsert logic.

    Args:
        connection: Database connection

        predictions_df: DataFrame containing predictions

        schema: Schema where the predictions table is located

        table: Table name for predictions

        primary_keys: List of columns to use as primary keys for upsert

    Returns:
        None
    """
    if primary_keys is None:
        primary_keys = ["home_team", "game_date"]

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
