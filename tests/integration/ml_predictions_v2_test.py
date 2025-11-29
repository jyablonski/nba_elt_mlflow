from sqlalchemy import text
import pandas as pd
import pytest

from src.utils import (
    pull_tonights_games,
    generate_win_predictions_v2,
    write_predictions_to_database,
)

SOURCE_SCHEMA = "silver"
DESTINATION_SCHEMA = "gold"
INPUT_TABLE_V2 = "ml_game_features_v2"
OUTPUT_TABLE_V2 = "ml_game_predictions_v2"


@pytest.fixture(autouse=True)
def cleanup_v2_predictions(postgres_conn):
    """Clean up V2 predictions table before and after each test."""
    # We assume the table exists thanks to the bootstrap script
    cleanup_sql = text(f"TRUNCATE TABLE {DESTINATION_SCHEMA}.{OUTPUT_TABLE_V2}")

    postgres_conn.execute(cleanup_sql)
    yield
    postgres_conn.execute(cleanup_sql)


def test_pull_tonights_games_v2(postgres_conn):
    """
    Test pulling V2 data from the database.
    Verifies that the new columns (e.g. home_active_vorp) are present.
    """
    games = pull_tonights_games(
        connection=postgres_conn, schema=SOURCE_SCHEMA, table=INPUT_TABLE_V2
    )

    assert not games.empty
    # Verify new V2 schema columns exist
    assert "home_active_vorp" in games.columns
    assert (
        "home_fatigue_index" not in games.columns
    )  # Should be raw, not engineered yet
    assert "home_travel_miles_last_7_days" in games.columns
    assert len(games) >= 8  # Based on your INSERT statement


def test_generate_win_predictions_v2(postgres_conn, v2_artifacts):
    """
    Test the full V2 inference pipeline:
    Raw Data -> FeatureEngineer (Transform) -> Model -> Predictions
    """
    # 1. Get Data
    games = pull_tonights_games(
        connection=postgres_conn, schema=SOURCE_SCHEMA, table=INPUT_TABLE_V2
    )

    # 2. Run Inference
    predictions = generate_win_predictions_v2(games_df=games, artifacts=v2_artifacts)

    # 3. Assertions
    assert not predictions.empty
    assert len(predictions) == len(games)

    # Check Logic
    assert predictions["home_team_predicted_win_pct"].between(0, 1).all()
    assert predictions["away_team_predicted_win_pct"].between(0, 1).all()
    # Sum to 1.0
    total_probs = (
        predictions["home_team_predicted_win_pct"]
        + predictions["away_team_predicted_win_pct"]
    )
    assert total_probs.round(3).eq(1.0).all()


def test_write_predictions_to_database_v2(postgres_conn, v2_artifacts):
    """Test writing V2 predictions to the new Gold table."""

    # 1. Generate
    games = pull_tonights_games(
        connection=postgres_conn, schema=SOURCE_SCHEMA, table=INPUT_TABLE_V2
    )
    predictions = generate_win_predictions_v2(games, v2_artifacts)

    # 2. Write
    write_predictions_to_database(
        connection=postgres_conn,
        predictions_df=predictions,
        schema=DESTINATION_SCHEMA,
        table=OUTPUT_TABLE_V2,
        primary_keys=["home_team", "game_date"],
    )

    # 3. Verify in DB
    result = pd.read_sql_query(
        f"SELECT * FROM {DESTINATION_SCHEMA}.{OUTPUT_TABLE_V2}",
        postgres_conn,
    )

    assert len(result) == len(predictions)
    # Check that timestamps were populated
    assert result["created_at"].notna().all()
    # Verify a specific row matches
    assert result.iloc[0]["home_team"] == predictions.iloc[0]["home_team"]
