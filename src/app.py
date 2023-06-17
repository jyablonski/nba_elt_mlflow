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

logging.info("STARTING NBA ELT MLFLOW Version: 1.5.0")

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

## 2022-05-01 - pasting this in bc you can alternatively read in the model from S3 rather than keep a local copy.

# import boto3
# import pickle

# s3 = boto3.resource('s3')
# clf = pickle.loads(s3.Bucket("jyablonski-mlflow-bucket").Object("8873275a193d4124ba5923568da6fa8e/artifacts/NBA_ELT_PIPELINE_MODEL/model.pkl").get()['Body'].read())

# this function performs the prediction and then joins the rest of the columns back in.
tonights_games_ml = calculate_win_pct(tonights_games, tonights_games_full, clf)

write_to_sql(conn, "tonights_games_ml", tonights_games_ml, "append")
