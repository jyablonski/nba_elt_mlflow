from joblib import load
import os

from jyablonski_common_modules.sql import create_sql_engine
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def postgres_conn():
    """Fixture to connect to Docker Postgres database"""
    # small override for local + docker testing to work fine
    if os.environ.get("ENV_TYPE") == "docker_dev":
        host = "postgres"
    else:
        host = "localhost"

    engine = create_sql_engine(
        user="postgres",
        password="postgres",
        host=host,
        database="postgres",
        schema="silver",
        port=5432,
    )

    with engine.begin() as conn:
        yield conn


@pytest.fixture(scope="session")
def ml_data():
    """
    Fixture to load player stats data from a csv file for testing.
    """
    fname = os.path.join(os.path.dirname(__file__), "fixtures/ml_df_test.csv")
    df = pd.read_csv(fname)
    return df


@pytest.fixture(scope="session")
def full_df():
    """
    Fixture to load player stats data from a csv file for testing.
    """
    fname = os.path.join(os.path.dirname(__file__), "fixtures/full_df_test.csv")
    df = pd.read_csv(fname)
    return df


@pytest.fixture(scope="session")
def ml_model():
    """
    Fixture to load player stats data from a csv file for testing.
    """
    fname = os.path.join(os.path.dirname(__file__), "fixtures/log_model.joblib")
    model = load(fname)
    return model


@pytest.fixture(scope="session")
def feature_flags_dataframe():
    """
    Fixture to load player stats data from a csv file for testing.
    """
    df = pd.DataFrame(data={"flag": ["season", "playoffs"], "is_enabled": [1, 0]})
    return df
