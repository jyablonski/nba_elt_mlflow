from joblib import load
import os

import pandas as pd
import pytest

@pytest.fixture(scope="session")
def ml_data():
    """
    Fixture to load player stats data from a csv file for testing.
    """
    fname = os.path.join(
        os.path.dirname(__file__), "fixtures/ml_df_test.csv"
    )
    df = pd.read_csv(fname)
    return df

@pytest.fixture(scope="session")
def full_df():
    """
    Fixture to load player stats data from a csv file for testing.
    """
    fname = os.path.join(
        os.path.dirname(__file__), "fixtures/full_df_test.csv"
    )
    df = pd.read_csv(fname)
    return df

@pytest.fixture(scope="session")
def ml_model():
    """
    Fixture to load player stats data from a csv file for testing.
    """
    fname = os.path.join(
        os.path.dirname(__file__), "fixtures/log_model.joblib"
    )
    model = load(fname)
    return model