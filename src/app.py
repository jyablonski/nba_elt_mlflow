import logging
from joblib import load
import sys

import pandas as pd

from src.utils import (
    calculate_win_pct,
    check_feature_flag,
    get_feature_flags,
    sql_connection,
    write_to_sql,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    handlers=[logging.StreamHandler()],
)

if __name__ == "__main__":
    logging.info("Starting NBA ELT MLFLOW Version: 1.6.7")

    conn = sql_connection("ml_models")
    with conn.begin() as connection:
        feature_flags = get_feature_flags(connection=connection)
        feature_flag_bool = check_feature_flag(flag="season", flags_df=feature_flags)

        if feature_flag_bool is False:
            logging.info("Season Feature Flag is disabled, exiting script ...")
            sys.exit(0)

        tonights_games_full = pd.read_sql_query(
            sql="select * from ml_tonights_games", con=connection
        ).sort_values("home_team_avg_pts_scored")
        log_regression_model = load("src/log_model.joblib")

        tonights_games_ml = calculate_win_pct(
            full_df=tonights_games_full, ml_model=log_regression_model
        )
        write_to_sql(con=connection, table_name="ml_game_predictions", df=tonights_games_ml, table_type="append")
    
    logging.info("Finished NBA ELT MLFLOW Version: 1.6.7")
