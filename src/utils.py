from datetime import datetime
import logging
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy.engine.base import Connection, Engine

INPUT_ML_TABLE = "ml_game_features"
OUTPUT_ML_TABLE = "ml_game_predictions"

ML_EXCLUDE_COLUMNS = [
    "home_team",
    "home_moneyline",
    "away_team",
    "away_moneyline",
    "game_date",
    "outcome",
]


def calculate_win_pct(
    full_df: pd.DataFrame,
    ml_model: LogisticRegression,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Calculate win prediction percentages for upcoming games.

    Args:
        full_df: DataFrame with game features and metadata
        ml_model: Trained logistic regression model
        logger: Optional logger instance

    Returns:
        DataFrame with predictions, or empty DataFrame if validation fails
    """
    log = logger or logging.getLogger(__name__)

    # Validation checks
    if full_df.empty:
        log.error("No data available for predictions")
        return pd.DataFrame()

    current_date = datetime.now().date()
    latest_date = pd.to_datetime(full_df["game_date"]).iloc[0].date()

    if latest_date != current_date:
        log.error(f"Date mismatch: data is from {latest_date}, expected {current_date}")
        return pd.DataFrame()

    # Prepare features for prediction
    ml_features = full_df.drop(columns=ML_EXCLUDE_COLUMNS)

    # Generate predictions
    predictions = ml_model.predict_proba(ml_features)
    prediction_df = pd.DataFrame(
        predictions,
        columns=["away_team_predicted_win_pct", "home_team_predicted_win_pct"],
    ).round(3)

    # Combine with original data
    result_df = full_df.reset_index(drop=True).drop(columns=["outcome"])
    result_df = pd.concat([result_df, prediction_df], axis=1)

    log.info(f"Generated predictions for {len(result_df)} games")
    return result_df


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
    logging.info(f"Retrieved {len(flags)} feature flags")
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
        logging.info(f"Feature flag '{flag}' not found")
        return False

    is_enabled = bool(flag_data["is_enabled"].iloc[0])

    if not is_enabled:
        logging.info(f"Feature flag '{flag}' is disabled")

    return is_enabled
