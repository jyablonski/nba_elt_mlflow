from datetime import datetime
from joblib import load

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.utils import calculate_win_pct

# assert that it exits out and returns 0 if date is not today's date.
def test_calculate_win_pct_exit(ml_data, full_df, ml_model):
    # 2022-04-13 - i made a change to only check for today's date, so just do this.
    df = calculate_win_pct(ml_data, full_df, ml_model)
    print(isinstance(df, list))

    assert len(df) == 0
    assert isinstance(df, list)

def test_calculate_win_pct(ml_data, full_df, ml_model):
    # 2022-04-13 - i made a change to only check for today's date, so just do this.
    full_df['proper_date'] = datetime.now().date()
    df = calculate_win_pct(ml_data, full_df, ml_model)

    assert len(df) == 8
    assert len(df.columns) == 18
    assert min(df['away_team_predicted_win_pct']) == 0.292
    assert max(df['home_team_predicted_win_pct']) == 0.708