from joblib import load

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.utils import calculate_win_pct

def test_calculate_win_pct(ml_data, full_df, ml_model):
    df = calculate_win_pct(ml_data, full_df, ml_model)

    assert len(df) == 8
    assert len(df.columns) == 18
    assert min(df['away_team_predicted_win_pct']) == 0.292
    assert max(df['home_team_predicted_win_pct']) == 0.708