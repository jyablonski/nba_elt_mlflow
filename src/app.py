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

logging.info("STARTING NBA ELT MLFLOW Version: 1.2.8")

conn = sql_connection("ml_models")

tonights_games_full = pd.read_sql_query(
    "select * from ml_tonights_games", conn
).sort_values(
    "home_team_avg_pts_scored"
)  # the full df

tonights_games = tonights_games_full.drop(
    ["home_team", "away_team", "proper_date", "outcome"], axis=1
)  # for ml

past_games = pd.read_sql_query("select * from ml_past_games", conn)
past_games_outcome = past_games["outcome"].to_numpy()
past_games = past_games.drop(
    ["home_team", "away_team", "proper_date", "outcome"], axis=1
).to_numpy()

logging.info(f"Loading Logistic Regression model")
clf = load('log_model.joblib')

tonights_ml = pd.DataFrame(clf.predict_proba(tonights_games)).rename(
    columns={0: "away_team_predicted_win_pct", 1: "home_team_predicted_win_pct"}
)

tonights_games_ml = tonights_games_full.reset_index().drop(
    "outcome", axis=1
)  # reset index so predictions match up correctly

tonights_games_ml["home_team_predicted_win_pct"] = tonights_ml[
    "home_team_predicted_win_pct"
].round(3)
tonights_games_ml["away_team_predicted_win_pct"] = tonights_ml[
    "away_team_predicted_win_pct"
].round(3)

logging.info(f"Predicted Win %s for {len(tonights_games_ml)} games")

tonights_games_ml.schema = 'Validated'

write_to_sql(conn, "tonights_games_ml", tonights_games_ml, "append")
