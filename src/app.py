import os
import sys

from joblib import load
from jyablonski_common_modules.logging import create_logger
from jyablonski_common_modules.sql import create_sql_engine, write_to_sql_upsert
import pandas as pd

from src.utils import (
    calculate_win_pct,
    check_feature_flag,
    get_feature_flags,
    INPUT_ML_TABLE,
    OUTPUT_ML_TABLE,
)

SOURCE_SCHEMA = "silver"
DESTINATION_SCHEMA = "gold"
MODEL_PATH = "src/log_model.joblib"


def main():
    """Main execution function for NBA ML predictions pipeline."""
    logger = create_logger()
    logger.info("Starting NBA ELT ML Pipeline")

    engine = create_sql_engine(
        user=os.environ.get("RDS_USER", "default"),
        password=os.environ.get("RDS_PW", "default"),
        host=os.environ.get("IP", "postgres"),
        database=os.environ.get("RDS_DB", "default"),
        schema=SOURCE_SCHEMA,
        port=int(os.environ.get("RDS_PORT", 5432)),
    )

    with engine.begin() as connection:
        # check feature flag
        feature_flags = get_feature_flags(
            connection=connection, schema=DESTINATION_SCHEMA
        )

        # exit if feature flag is disabled
        if not check_feature_flag(flag="season", flags_df=feature_flags):
            logger.info("Season feature flag is disabled, exiting script")
            sys.exit(0)

        # Load data and model
        tonights_games = pd.read_sql_query(
            sql=f"select * from {SOURCE_SCHEMA}.{INPUT_ML_TABLE}", con=connection
        ).sort_values("home_team_avg_pts_scored")

        if tonights_games.empty:
            logger.warning("No games found for prediction, exiting script")
            sys.exit(0)

        model = load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")

        # generate predictions
        predictions = calculate_win_pct(
            full_df=tonights_games, ml_model=model, logger=logger
        )

        # exit if there are no predictions, or no games today
        if predictions.empty:
            logger.warning("No predictions generated, exiting script")
            sys.exit(0)

        # write predictions to database
        write_to_sql_upsert(
            conn=connection,
            table=OUTPUT_ML_TABLE,
            schema=DESTINATION_SCHEMA,
            df=predictions,
            primary_keys=["home_team", "game_date"],
        )
        logger.info(
            f"Successfully wrote {len(predictions)} predictions to {DESTINATION_SCHEMA}.{OUTPUT_ML_TABLE}"
        )

    logger.info("Finished NBA ELT ML Pipeline")


if __name__ == "__main__":
    main()
