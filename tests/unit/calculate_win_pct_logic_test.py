from datetime import datetime

import pandas as pd

from src.utils import calculate_win_pct


def test_calculate_win_pct_exit(full_df, ml_model):
    full_df["game_date"] = "2021-01-01"
    df = calculate_win_pct(full_df=full_df, ml_model=ml_model)

    assert len(df) == 0
    assert isinstance(df, pd.DataFrame)


def test_calculate_win_pct_values(full_df, ml_model):
    full_df["game_date"] = datetime.utcnow().date()
    df = calculate_win_pct(full_df=full_df, ml_model=ml_model)

    assert len(df) == 2
    assert len(df.columns) == 22
    assert min(df["away_team_predicted_win_pct"]) == 0.265
    assert max(df["home_team_predicted_win_pct"]) == 0.735
