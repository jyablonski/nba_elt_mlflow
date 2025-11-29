from unittest.mock import MagicMock
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
    assert "home_active_vorp" in games.columns
    assert "home_fatigue_index" not in games.columns
    assert "home_travel_miles_last_7_days" in games.columns
    assert len(games) >= 8


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
    assert predictions["home_team_predicted_win_pct"].between(0, 1).all()
    assert predictions["away_team_predicted_win_pct"].between(0, 1).all()

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
    assert result["created_at"].notna().all()
    assert result.iloc[0]["home_team"] == predictions.iloc[0]["home_team"]


def test_write_predictions_to_database_no_rows(postgres_conn, v2_artifacts):
    """Test writing V2 predictions to the new Gold table."""

    # 1. Generate
    games = pull_tonights_games(
        connection=postgres_conn, schema=SOURCE_SCHEMA, table=INPUT_TABLE_V2
    )
    predictions = generate_win_predictions_v2(games, v2_artifacts)

    predictions = predictions.iloc[0:0]  # Empty the DataFrame

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
    assert result.empty


def test_generate_v2_date_mismatch(v2_input_data, v2_artifacts):
    """
    Cover the branch: if latest_date != current_date
    """
    # Set date to the past
    v2_input_data["game_date"] = "2020-01-01"

    # Run prediction (should proceed anyway, but log a warning)
    # We just need to ensure it runs without error to cover the line
    results = generate_win_predictions_v2(v2_input_data, v2_artifacts)

    assert not results.empty
    assert len(results) == len(v2_input_data)


def test_generate_v2_missing_feature_column(v2_input_data, v2_artifacts):
    """
    Cover the branch: if col not in X_processed.columns: X_processed[col] = 0

    We drop a REQUIRED column (e.g., 'home_team_rank') from the input.
    The code should notice it's missing, fill it with 0s, and the model
    will run successfully because the column exists again.
    """
    # 1. Corrupt the input data by removing a known feature
    # The FeatureEngineer normally passes this through, so if we drop it here,
    # it will be missing from X_processed.
    corrupt_data = v2_input_data.drop(columns=["home_team_rank"])

    # 2. Run prediction
    results = generate_win_predictions_v2(corrupt_data, v2_artifacts)

    # 3. Assert success
    # If the logic failed, the model would error on "missing column" or pandas would error
    assert not results.empty
    assert len(results) == len(v2_input_data)


def test_generate_v2_outcome_drop(v2_input_data, v2_artifacts):
    """
    Cover the branch: if "outcome" in result_df.columns: result_df.drop(...)
    """
    # Add outcome column
    v2_input_data["outcome"] = "W"

    results = generate_win_predictions_v2(v2_input_data, v2_artifacts)

    # Ensure it was dropped in the result
    assert "outcome" not in results.columns


def test_generate_v2_feature_engineering_error(v2_input_data, v2_artifacts):
    """
    Cover the first try/except block.
    We mock the engineer to raise an exception.
    """
    # Create a mock that raises an error when preprocess_data is called
    mock_engineer = MagicMock()
    mock_engineer.preprocess_data.side_effect = Exception("Boom!")

    # Inject the mock into artifacts
    broken_artifacts = v2_artifacts.copy()
    broken_artifacts["feature_engineer"] = mock_engineer

    results = generate_win_predictions_v2(v2_input_data, broken_artifacts)

    # Should catch error, log it, and return empty DF
    assert results.empty
    assert isinstance(results, pd.DataFrame)


def test_generate_v2_model_prediction_error(v2_input_data, v2_artifacts):
    """
    Cover the second try/except block.
    We mock the model to raise an exception.
    """
    # Create a mock that raises an error when predict_proba is called
    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = Exception("Model Exploded!")

    # Inject the mock into artifacts
    broken_artifacts = v2_artifacts.copy()
    broken_artifacts["model"] = mock_model

    results = generate_win_predictions_v2(v2_input_data, broken_artifacts)

    # Should catch error, log it, and return empty DF
    assert results.empty
