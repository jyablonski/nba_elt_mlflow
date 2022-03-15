import datetime
import logging
from joblib import load

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import *

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    handlers=[logging.FileHandler("logs/example.log"), logging.StreamHandler()],
)

logging.info("STARTING NBA ELT MLFLOW Version: 1.3.2")

conn = sql_connection("ml_models")

tonights_games_full = pd.read_sql_query(
    "select * from ml_tonights_games", conn
).sort_values(
    "home_team_avg_pts_scored"
)  # the full df

tonights_games = tonights_games_full.drop(
    ["home_team", "away_team", "proper_date", "outcome"], axis=1
)  # for ml

logging.info(f"Loading Logistic Regression model")
clf = load("log_model.joblib")

tonights_games_ml = calculate_win_pct(tonights_games, tonights_games_full, clf)

write_to_sql(conn, "tonights_games_ml", tonights_games_ml, "append")
