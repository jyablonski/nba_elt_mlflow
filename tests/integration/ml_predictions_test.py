from sqlalchemy import text
import pandas as pd
import pytest

from src.utils import (
    pull_tonights_games,
    generate_win_predictions,
    write_predictions_to_database,
    load_ml_model,
    OUTPUT_ML_TABLE,
)

SOURCE_SCHEMA = "silver"
DESTINATION_SCHEMA = "gold"


@pytest.fixture(autouse=True)
def cleanup_predictions(postgres_conn):
    """Clean up predictions table before and after each test."""
    cleanup_sql = text(f"TRUNCATE TABLE {DESTINATION_SCHEMA}.{OUTPUT_ML_TABLE}")

    postgres_conn.execute(cleanup_sql)
    yield
    postgres_conn.execute(cleanup_sql)


def test_pull_tonights_games(postgres_conn):
    """Test pulling tonight's games from database."""
    games = pull_tonights_games(connection=postgres_conn, schema=SOURCE_SCHEMA)

    assert not games.empty, "Should retrieve games"
    assert "home_team" in games.columns
    assert "away_team" in games.columns
    assert "game_date" in games.columns


def test_load_ml_model():
    """Test loading ML model from disk."""
    model = load_ml_model(model_path="tests/fixtures/log_model.joblib")

    assert model is not None
    assert hasattr(model, "predict_proba")


def test_generate_win_predictions(postgres_conn, ml_model):
    """Test generating win predictions from game features."""
    games = pull_tonights_games(connection=postgres_conn, schema=SOURCE_SCHEMA)

    predictions = generate_win_predictions(games_df=games, ml_model=ml_model)

    assert len(predictions) == 4
    assert "home_team_predicted_win_pct" in predictions.columns
    assert "away_team_predicted_win_pct" in predictions.columns

    # verify probabilities are between 0 and 1
    assert all(predictions["home_team_predicted_win_pct"].between(0, 1))
    assert all(predictions["away_team_predicted_win_pct"].between(0, 1))


def test_write_predictions_to_database(postgres_conn, ml_model):
    """Test writing predictions to database."""
    games = pull_tonights_games(connection=postgres_conn, schema=SOURCE_SCHEMA)

    predictions = generate_win_predictions(games_df=games, ml_model=ml_model)

    write_predictions_to_database(
        connection=postgres_conn, predictions_df=predictions, schema=DESTINATION_SCHEMA
    )

    # Verify data was written
    result = pd.read_sql_query(
        f"SELECT COUNT(*) as count FROM {DESTINATION_SCHEMA}.{OUTPUT_ML_TABLE}",
        postgres_conn,
    )

    assert result["count"].iloc[0] == 4


def test_generate_win_predictions_empty_dataframe(ml_model):
    """Test handling of empty DataFrame."""
    empty_df = pd.DataFrame()
    result = generate_win_predictions(games_df=empty_df, ml_model=ml_model)

    assert result.empty


def test_full_pipeline_integration(postgres_conn, ml_model):
    """Integration test for the complete pipeline."""
    # Pull games
    games = pull_tonights_games(connection=postgres_conn, schema=SOURCE_SCHEMA)

    # Generate predictions
    predictions = generate_win_predictions(games_df=games, ml_model=ml_model)

    # Write to database
    write_predictions_to_database(
        connection=postgres_conn, predictions_df=predictions, schema=DESTINATION_SCHEMA
    )

    # Verify end-to-end
    db_predictions = pd.read_sql_query(
        f"SELECT * FROM {DESTINATION_SCHEMA}.{OUTPUT_ML_TABLE}", postgres_conn
    )

    assert len(db_predictions) == len(predictions)
    assert all(db_predictions["home_team_predicted_win_pct"].notna())
