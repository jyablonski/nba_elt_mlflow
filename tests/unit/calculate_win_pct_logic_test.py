from datetime import datetime

from src.utils import calculate_win_pct


def test_calculate_win_pct_exit(ml_data, full_df, ml_model):
    full_df["proper_date"] = "2021-01-01"
    df = calculate_win_pct(ml_data, full_df, ml_model)

    assert len(df) == 0
    assert isinstance(df, list)


def test_calculate_win_pct_values(ml_data, full_df, ml_model):
    full_df["proper_date"] = datetime.now().date()
    df = calculate_win_pct(ml_data, full_df, ml_model)

    assert len(df) == 2
    assert len(df.columns) == 22
    assert min(df["away_team_predicted_win_pct"]) == 0.265
    assert max(df["home_team_predicted_win_pct"]) == 0.735
