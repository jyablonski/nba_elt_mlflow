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
    Assumes tables are pre-loaded via bootstrap script.
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
def v2_artifacts():
    """
    Load V2 Production Artifacts (Dict containing Model + Feature Engineer).
    Assumes you have copied the generated model to tests/fixtures/log_model_v2.joblib
    """
    path = FIXTURES_DIR / "log_model_v2.joblib"
    if not path.exists():
        pytest.skip("V2 Model artifact not found in fixtures")
    return load(path)


@pytest.fixture(scope="session")
def feature_flags_dataframe() -> pd.DataFrame:
    """Create sample feature flags DataFrame for testing."""
    return pd.DataFrame({"flag": ["season", "playoffs"], "is_enabled": [1, 0]})


@pytest.fixture
def v2_input_data() -> pd.DataFrame:
    """
    Creates a sample V2 DataFrame for testing inference.
    Contains all columns required by the V2 Feature Schema.
    """
    data = {
        # Metadata
        "home_team": ["Phoenix Suns"],
        "away_team": ["Denver Nuggets"],
        "game_date": [
            pd.Timestamp.now().date()
        ],  # Defaults to today for date validation tests
        "home_moneyline": [-150],
        "away_moneyline": [130],
        # Team Stats (Win Pct, Rank, Scoring)
        "home_team_rank": [5],
        "away_team_rank": [3],
        "home_team_win_pct": [0.650],
        "away_team_win_pct": [0.700],
        "home_team_win_pct_last10": [0.800],
        "away_team_win_pct_last10": [0.600],
        "home_team_avg_pts_scored": [115.5],
        "away_team_avg_pts_scored": [112.0],
        "home_team_avg_pts_scored_opp": [110.0],
        "away_team_avg_pts_scored_opp": [108.0],
        # V2 Specifics (Fatigue / Travel)
        "home_days_rest": [1],
        "away_days_rest": [0],  # Back-to-back
        "home_games_last_7_days": [3],
        "away_games_last_7_days": [4],
        "home_travel_miles_last_7_days": [500.0],
        "away_travel_miles_last_7_days": [1200.0],
        "home_is_cross_country_trip": [0],
        "away_is_cross_country_trip": [0],
        # V2 Specifics (Talent / VORP)
        "home_star_score": [3],
        "away_star_score": [2],
        "home_active_vorp": [2.5],
        "away_active_vorp": [4.2],
        "home_pct_vorp_missing": [0.0],
        "away_pct_vorp_missing": [0.15],
    }
    return pd.DataFrame(data)
