import os
from pathlib import Path

from joblib import load
from jyablonski_common_modules.sql import create_sql_engine
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_DB_USER = "postgres"
TEST_DB_PASSWORD = "postgres"
TEST_DB_NAME = "postgres"
TEST_DB_PORT = 5432


@pytest.fixture(scope="session")
def postgres_conn():
    """
    Fixture to connect to Docker Postgres database.

    Uses localhost for local testing, 'postgres' hostname for Docker Compose.
    """
    host = "postgres" if os.environ.get("ENV_TYPE") == "docker_dev" else "localhost"

    engine = create_sql_engine(
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        host=host,
        database=TEST_DB_NAME,
        schema="silver",
        port=TEST_DB_PORT,
    )

    with engine.begin() as conn:
        yield conn


@pytest.fixture(scope="session")
def ml_data() -> pd.DataFrame:
    """Load ML test data from CSV fixture."""
    return pd.read_csv(FIXTURES_DIR / "ml_df_test.csv")


@pytest.fixture(scope="session")
def full_df() -> pd.DataFrame:
    """Load full DataFrame test data from CSV fixture."""
    return pd.read_csv(FIXTURES_DIR / "full_df_test.csv")


@pytest.fixture(scope="session")
def ml_model():
    """Load trained ML model from joblib fixture."""
    return load(FIXTURES_DIR / "log_model.joblib")


@pytest.fixture(scope="session")
def feature_flags_dataframe() -> pd.DataFrame:
    """Create sample feature flags DataFrame for testing."""
    return pd.DataFrame({"flag": ["season", "playoffs"], "is_enabled": [1, 0]})
