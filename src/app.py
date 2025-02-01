from joblib import load
import os
import sys

from jyablonski_common_modules.logging import create_logger
from jyablonski_common_modules.sql import create_sql_engine, write_to_sql_upsert
import pandas as pd

from src.utils import (
    calculate_win_pct,
    check_feature_flag,
    get_feature_flags,
)

if __name__ == "__main__":
    logger = create_logger()
    logger.info("Starting NBA ELT MLFLOW Version: 1.7.0")
    ml_schema = "ml"

    engine = create_sql_engine(
        user=os.environ.get("RDS_USER", default="default"),
        password=os.environ.get("RDS_PW", default="default"),
        host=os.environ.get("IP", "postgres"),
        database=os.environ.get("RDS_DB", default="default"),
        schema=ml_schema,
        port=os.environ.get("RDS_PORT", default=5432),
    )
    with engine.begin() as connection:
        feature_flags = get_feature_flags(connection=connection)
        feature_flag_bool = check_feature_flag(flag="season", flags_df=feature_flags)

        if not feature_flag_bool:
            logger.info("Season Feature Flag is disabled, exiting script ...")
            sys.exit(0)

        tonights_games_full = pd.read_sql_query(
            sql="select * from ml_tonights_games", con=connection
        ).sort_values("home_team_avg_pts_scored")
        log_regression_model = load("src/log_model.joblib")

        tonights_games_ml = calculate_win_pct(
            full_df=tonights_games_full, ml_model=log_regression_model
        )

        write_to_sql_upsert(
            conn=connection,
            table="ml_game_predictions",
            schema=ml_schema,
            df=tonights_games_ml,
            primary_keys=["home_team", "game_date"],
        )

    logger.info("Finished NBA ELT MLFLOW Version: 1.7.0")
