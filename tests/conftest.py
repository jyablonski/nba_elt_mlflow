from pathlib import Path

import pandas as pd
import pytest
from joblib import load
from jyablonski_common_modules.sql import create_sql_engine
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from testcontainers.postgres import PostgresContainer

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SQL = PROJECT_ROOT / "docker" / "postgres_bootstrap.sql"


def pytest_collection_modifyitems(items):
    """Apply unit/integration markers from test directory layout."""
    for item in items:
        path = Path(str(item.fspath))
        if "integration" in path.parts:
            item.add_marker(pytest.mark.integration)
        elif "unit" in path.parts:
            item.add_marker(pytest.mark.unit)


def _bootstrap_database(engine) -> None:
    sql = BOOTSTRAP_SQL.read_text()
    conn = engine.raw_connection()
    try:
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cursor:
            cursor.execute(sql)
    finally:
        conn.close()


@pytest.fixture(scope="session")
def postgres_container():
    """Ephemeral Postgres for integration tests (requires Docker)."""
    with PostgresContainer(
        "postgres:16-alpine",
        username="postgres",
        password="postgres",
        dbname="postgres",
    ) as postgres:
        yield postgres


@pytest.fixture(scope="session")
def postgres_engine(postgres_container):
    engine = create_sql_engine(
        user=postgres_container.username,
        password=postgres_container.password,
        host=postgres_container.get_container_host_ip(),
        database=postgres_container.dbname,
        schema="silver",
        port=int(postgres_container.get_exposed_port(5432)),
    )
    _bootstrap_database(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def postgres_conn(postgres_engine):
    """
    Session connection to the Testcontainers Postgres database.
    Schema and seed data are loaded from docker/postgres_bootstrap.sql.
    """
    with postgres_engine.begin() as conn:
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
        pytest.skip("V2 Model artifact not found in fixtures")  # ty: ignore[too-many-positional-arguments]
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
