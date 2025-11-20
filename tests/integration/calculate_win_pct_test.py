from jyablonski_common_modules.sql import write_to_sql_upsert
import pandas as pd
import pytest
from sqlalchemy import text

from src.utils import calculate_win_pct, INPUT_ML_TABLE, OUTPUT_ML_TABLE

# Constants
SOURCE_SCHEMA = "silver"
DESTINATION_SCHEMA = "gold"


@pytest.fixture(autouse=True)
def cleanup_predictions(postgres_conn):
    """Clean up predictions table before and after each test."""
    cleanup_sql = text(f"TRUNCATE TABLE {DESTINATION_SCHEMA}.{OUTPUT_ML_TABLE}")

    # Cleanup before test
    postgres_conn.execute(cleanup_sql)

    yield

    # Cleanup after test
    postgres_conn.execute(cleanup_sql)


def test_calculate_win_pct_postgres(postgres_conn, ml_model):
    """
    Test the full ML prediction pipeline:
    1. Load game features from database
    2. Generate predictions
    3. Write predictions to database
    4. Verify results
    """
    # Load input data
    tonights_games = pd.read_sql_query(
        f"SELECT * FROM {SOURCE_SCHEMA}.{INPUT_ML_TABLE}", postgres_conn
    ).sort_values("home_team_avg_pts_scored")

    # Generate predictions
    predictions = calculate_win_pct(full_df=tonights_games, ml_model=ml_model)

    # Write to database
    write_to_sql_upsert(
        conn=postgres_conn,
        table=OUTPUT_ML_TABLE,
        schema=DESTINATION_SCHEMA,
        df=predictions,
        primary_keys=["home_team", "game_date"],
    )

    # Verify results
    result_count = pd.read_sql_query(
        f"SELECT COUNT(*) as count FROM {DESTINATION_SCHEMA}.{OUTPUT_ML_TABLE}",
        postgres_conn,
    )

    assert len(predictions) == 4, "Should generate 4 predictions"
    assert result_count["count"].iloc[0] == 4, "Should have 4 records in database"

    # Verify predictions have expected columns
    expected_columns = [
        "home_team",
        "away_team",
        "game_date",
        "home_team_predicted_win_pct",
        "away_team_predicted_win_pct",
    ]
    for col in expected_columns:
        assert col in predictions.columns, f"Missing expected column: {col}"


def test_calculate_win_pct_with_empty_dataframe(ml_model):
    """Test that calculate_win_pct handles empty DataFrames gracefully."""
    empty_df = pd.DataFrame()
    result = calculate_win_pct(full_df=empty_df, ml_model=ml_model)

    assert result.empty, "Should return empty DataFrame for empty input"


def test_predictions_probabilities_sum_to_one(postgres_conn, ml_model):
    """Test that home and away win probabilities sum to approximately 1.0."""
    tonights_games = pd.read_sql_query(
        f"SELECT * FROM {SOURCE_SCHEMA}.{INPUT_ML_TABLE}", postgres_conn
    )

    predictions = calculate_win_pct(full_df=tonights_games, ml_model=ml_model)

    probability_sums = (
        predictions["home_team_predicted_win_pct"]
        + predictions["away_team_predicted_win_pct"]
    )

    # Allow small floating point errors
    assert all(abs(probability_sums - 1.0) < 0.01), (
        "Win probabilities should sum to approximately 1.0"
    )
