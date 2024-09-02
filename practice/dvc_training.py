import os
from joblib import dump

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# one hot encoding is not necessary for `is_top_players`
# 0 < 1 < 2 - it actually has ordinal value
# if you asked ppl what their fave color was and used 1 for red, 2 for blue, 3 for green
# etc then that would need to be one hot encoded because the differential in value
# has no meaningful significance.

# conn = sql_connection('ml_models')
# past_games = pd.read_sql_query('select * from ml_past_games_odds_analysis;', conn).query(f"outcome.notna()")
past_games = pd.read_csv("data/past_games_2023-10-18.csv")
past_games["outcome"] = past_games["outcome"].replace({"W": 1, "L": 0})

past_games_outcome = past_games["outcome"]

past_games_ml_dataset = past_games.drop(
    [
        "home_team_predicted_win_pct",
        "away_team_predicted_win_pct",
        "ml_accuracy",
        "ml_money_col",
        "home_implied_probability",
        "away_implied_probability",
        "outcome",
        "ml_prediction",
        "actual_outcome",
        "proper_date",
        "away_team",
        "home_team",
    ],
    axis=1,
)

## my old way
past_games_outcome = past_games_outcome.to_numpy()
training_set = past_games_ml_dataset.to_numpy()

clf_linear_svc = LinearSVC(random_state=0).fit(training_set, past_games_outcome)
clf_svc = SVC(random_state=0).fit(training_set, past_games_outcome)
clf = LogisticRegression(random_state=0).fit(training_set, past_games_outcome)


directory = 'data/first_test'
filename = 'past_games_2023-10-18.joblib'
full_path = os.path.join(directory, filename)

if not os.path.exists(directory):
    os.makedirs(directory)

dump(value=clf, filename=full_path)