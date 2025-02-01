from datetime import datetime
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy.engine.base import Connection, Engine


def calculate_win_pct(
    full_df: pd.DataFrame, ml_model: LogisticRegression
) -> pd.DataFrame:
    """
    Function to Calculate Win Prediction %s for Upcoming Games

    Args:
        full_df (pd.DataFrame): The Full DataFrame with other variables not
            needed for the ML Predictions

        ml_model (LogisticRegression): The ML Model loaded from
            `log_model.joblib`

    Returns:
        Pandas DataFrame of ML Predictions to append into `ml_game_predictions`
    """
    if full_df.empty:
        logging.error("Exiting out, no data for today's games.")
        return pd.DataFrame()

    current_date = datetime.now().date()
    latest_date = pd.to_datetime(full_df["game_date"].drop_duplicates().iloc[0]).date()

    if latest_date != current_date:
        logging.error(f"Exiting out, {latest_date} != {current_date}")
        return pd.DataFrame()

    # Drop unnecessary columns for prediction
    ml_df = full_df.drop(
        columns=[
            "home_team",
            "home_moneyline",
            "away_team",
            "away_moneyline",
            "game_date",
            "outcome",
        ]
    )

    # Predict win probabilities
    predictions = ml_model.predict_proba(ml_df)
    df_preds = pd.DataFrame(
        predictions,
        columns=["away_team_predicted_win_pct", "home_team_predicted_win_pct"],
    ).round(3)

    # Merge predictions back into the original DataFrame
    df_final = full_df.reset_index(drop=True).drop(columns=["outcome"])
    df_final = df_final.assign(**df_preds)

    logging.info(f"Predicted Win %s for {len(df_final)} games")
    return df_final


def get_feature_flags(connection: Connection | Engine) -> pd.DataFrame:
    flags = pd.read_sql_query(sql="select * from marts.feature_flags;", con=connection)

    logging.info(f"Retrieving {len(flags)} Feature Flags")
    return flags


def check_feature_flag(flag: str, flags_df: pd.DataFrame) -> bool:
    flags_df = flags_df.query(f"flag == '{flag}'")

    if len(flags_df) > 0 and flags_df["is_enabled"].iloc[0] == 1:
        return True
    else:
        print(f"Feature Flag for {flag} is disabled, skipping")
        logging.info(f"Feature Flag for {flag} is disabled, skipping")
        return False
