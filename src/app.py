import logging
from joblib import load
import sys

import pandas as pd

from utils import *

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    handlers=[logging.FileHandler("logs/example.log"), logging.StreamHandler()],
)

logging.info("STARTING NBA ELT MLFLOW Version: 1.5.1")

conn = sql_connection("ml_models")

feature_flags = get_feature_flags(conn)

feature_flag_bool = check_feature_flag(flag="season", flags_df=feature_flags)

if feature_flag_bool == False:
    logging.info(f"Season Feature Flag is disabled, exiting script ...")
    sys.exit(0)


tonights_games_full = pd.read_sql_query(
    "select * from ml_tonights_games", conn
).sort_values(
    "home_team_avg_pts_scored"
)  # the full df

tonights_games = tonights_games_full.drop(
    [
        "home_team",
        "home_moneyline",
        "away_team",
        "away_moneyline",
        "proper_date",
        "outcome",
    ],
    axis=1,
)  # i'm just dropping every column not used in ml.

logging.info(f"Loading Logistic Regression model")
clf = load("log_model.joblib")

tonights_games_ml = calculate_win_pct(tonights_games, tonights_games_full, clf)

write_to_sql(conn, "tonights_games_ml", tonights_games_ml, "append")
