import pandas as pd
import pytest
from sqlalchemy import text

from src import app
from src.app import ExitCode
from src.utils import OUTPUT_ML_TABLE

SOURCE_SCHEMA = "silver"
DESTINATION_SCHEMA = "gold"
INPUT_TABLES = [
    (SOURCE_SCHEMA, "ml_game_features"),
    (DESTINATION_SCHEMA, "feature_flags"),
]


@pytest.fixture
def pipeline_env(postgres_engine, postgres_container, monkeypatch):
    """
    Point app._create_db_engine at the Testcontainers Postgres and disable
    external side effects: an unset DASHBOARD_REFRESH_URL makes the dashboard
    refresh a no-op, and an unset webhook makes Slack alerts a no-op.
    """
    monkeypatch.setenv("RDS_USER", postgres_container.username)
    monkeypatch.setenv("RDS_PW", postgres_container.password)
    monkeypatch.setenv("IP", postgres_container.get_container_host_ip())
    monkeypatch.setenv("RDS_DB", postgres_container.dbname)
    monkeypatch.setenv("RDS_PORT", str(postgres_container.get_exposed_port(5432)))
    monkeypatch.delenv("DASHBOARD_REFRESH_URL", raising=False)
    monkeypatch.delenv(app.SLACK_WEBHOOK_ENV, raising=False)
    return postgres_engine


@pytest.fixture(autouse=True)
def restore_seed_data(postgres_engine):
    """
    run_pipeline commits real writes and individual tests mutate the seeded
    input tables, so snapshot the inputs up front and restore them afterwards.
    """
    snapshots = {}
    with postgres_engine.connect() as conn:
        for schema, table in INPUT_TABLES:
            snapshots[(schema, table)] = pd.read_sql_query(
                text(f"SELECT * FROM {schema}.{table}"), conn
            )

    yield

    with postgres_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {DESTINATION_SCHEMA}.{OUTPUT_ML_TABLE}"))
        for (schema, table), snapshot in snapshots.items():
            conn.execute(text(f"TRUNCATE TABLE {schema}.{table}"))
            snapshot.to_sql(
                table, conn, schema=schema, if_exists="append", index=False
            )


def _prediction_count(engine) -> int:
    with engine.connect() as conn:
        result = pd.read_sql_query(
            text(
                f"SELECT COUNT(*) AS count FROM {DESTINATION_SCHEMA}.{OUTPUT_ML_TABLE}"
            ),
            conn,
        )
    return int(result["count"].iloc[0])


def test_run_pipeline_success(pipeline_env):
    """Season enabled with games for today: predictions are written and committed."""
    engine = pipeline_env

    assert app.run_pipeline() is ExitCode.SUCCESS

    with engine.connect() as conn:
        predictions = pd.read_sql_query(
            text(f"SELECT * FROM {DESTINATION_SCHEMA}.{OUTPUT_ML_TABLE}"), conn
        )

    assert len(predictions) == 4
    assert predictions["home_team_predicted_win_pct"].between(0, 1).all()
    assert predictions["away_team_predicted_win_pct"].between(0, 1).all()


def test_run_pipeline_skips_when_season_flag_disabled(pipeline_env):
    """Season flag off: pipeline exits cleanly without writing predictions."""
    engine = pipeline_env
    with engine.begin() as conn:
        conn.execute(
            text(
                f"UPDATE {DESTINATION_SCHEMA}.feature_flags "
                "SET is_enabled = 0 WHERE flag = 'season'"
            )
        )

    assert app.run_pipeline() is ExitCode.SUCCESS
    assert _prediction_count(engine) == 0


def test_run_pipeline_skips_when_no_games(pipeline_env):
    """No games available: pipeline exits cleanly without writing predictions."""
    engine = pipeline_env
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {SOURCE_SCHEMA}.ml_game_features"))

    assert app.run_pipeline() is ExitCode.SUCCESS
    assert _prediction_count(engine) == 0


def test_run_pipeline_skips_when_predictions_empty(pipeline_env):
    """
    Games are stale (not today), so generate_win_predictions returns empty and
    the pipeline exits cleanly without writing predictions.
    """
    engine = pipeline_env
    with engine.begin() as conn:
        conn.execute(
            text(
                f"UPDATE {SOURCE_SCHEMA}.ml_game_features "
                "SET game_date = current_date - INTERVAL '7 days'"
            )
        )

    assert app.run_pipeline() is ExitCode.SUCCESS
    assert _prediction_count(engine) == 0


def test_run_pipeline_returns_failure_on_db_error(pipeline_env, monkeypatch):
    """An unreachable database surfaces as an operational failure (exit code 1)."""
    monkeypatch.setenv("RDS_PORT", "1")  # connection refused on loopback

    assert app.run_pipeline() is ExitCode.FAILURE
    assert _prediction_count(pipeline_env) == 0
