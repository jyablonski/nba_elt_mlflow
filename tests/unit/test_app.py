from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src import app


@pytest.fixture
def mock_engine():
    engine = MagicMock(name="engine")
    connection = MagicMock(name="connection")

    @contextmanager
    def connect_ctx():
        yield connection

    @contextmanager
    def begin_ctx():
        yield connection

    engine.connect.return_value = connect_ctx()
    engine.begin.return_value = begin_ctx()
    return engine


@pytest.fixture
def sample_games():
    return pd.DataFrame(
        {
            "home_team": ["Boston Celtics"],
            "away_team": ["Chicago Bulls"],
            "game_date": [pd.Timestamp.now().date()],
            "home_moneyline": [-160],
            "away_moneyline": [200],
            "home_team_rank": [8],
            "home_days_rest": [3],
            "home_team_avg_pts_scored": [116.3],
            "home_team_avg_pts_scored_opp": [112.4],
            "home_team_win_pct": [0.75],
            "home_team_win_pct_last10": [0.63],
            "home_is_top_players": [2],
            "away_team_rank": [14],
            "away_days_rest": [0],
            "away_team_avg_pts_scored": [115.3],
            "away_team_avg_pts_scored_opp": [117.2],
            "away_team_win_pct": [0.45],
            "away_team_win_pct_last10": [0.48],
            "away_is_top_players": [2],
        }
    )


@pytest.fixture
def sample_predictions(sample_games):
    preds = sample_games.copy()
    preds["home_team_predicted_win_pct"] = 0.6
    preds["away_team_predicted_win_pct"] = 0.4
    return preds


@patch("src.app._create_db_engine")
@patch("src.app.get_feature_flags", return_value=pd.DataFrame())
@patch("src.app.check_feature_flag", return_value=False)
def test_run_pipeline_exits_when_season_flag_disabled(
    _check_flag,
    _get_flags,
    mock_create_engine,
    mock_engine,
):
    mock_create_engine.return_value = mock_engine

    assert app.run_pipeline() == 0
    mock_engine.dispose.assert_called_once()


@patch("src.app._create_db_engine")
@patch("src.app.get_feature_flags", return_value=pd.DataFrame())
@patch("src.app.check_feature_flag", return_value=True)
@patch("src.app.pull_tonights_games", return_value=pd.DataFrame())
@patch("src.app.load_ml_model")
def test_run_pipeline_exits_when_no_games(
    mock_load_model,
    _pull_games,
    _check_flag,
    _get_flags,
    mock_create_engine,
    mock_engine,
):
    mock_create_engine.return_value = mock_engine

    assert app.run_pipeline() == 0
    mock_load_model.assert_not_called()


@patch("src.app._create_db_engine")
@patch("src.app.get_feature_flags", return_value=pd.DataFrame())
@patch("src.app.check_feature_flag", return_value=True)
@patch("src.app.pull_tonights_games")
@patch("src.app.load_ml_model")
@patch("src.app.generate_win_predictions", return_value=pd.DataFrame())
@patch("src.app.write_predictions_to_database")
def test_run_pipeline_exits_when_predictions_empty(
    mock_write,
    _generate,
    _load_model,
    mock_pull_games,
    _check_flag,
    _get_flags,
    mock_create_engine,
    mock_engine,
    sample_games,
):
    mock_create_engine.return_value = mock_engine
    mock_pull_games.return_value = sample_games

    assert app.run_pipeline() == 0
    mock_write.assert_not_called()


@patch("src.app._notify_slack_error")
@patch("src.app.run_dashboard_post_pipeline_checks")
@patch("src.app.write_predictions_to_database")
@patch("src.app.generate_win_predictions")
@patch("src.app.load_ml_model", return_value=MagicMock())
@patch("src.app.pull_tonights_games")
@patch("src.app.check_feature_flag", return_value=True)
@patch("src.app.get_feature_flags", return_value=pd.DataFrame())
@patch("src.app._create_db_engine")
def test_run_pipeline_success(
    mock_create_engine,
    _get_flags,
    _check_flag,
    mock_pull_games,
    _load_model,
    mock_generate,
    mock_write,
    mock_refresh,
    mock_slack,
    mock_engine,
    sample_games,
    sample_predictions,
):
    mock_create_engine.return_value = mock_engine
    mock_pull_games.return_value = sample_games
    mock_generate.return_value = sample_predictions

    assert app.run_pipeline() == 0
    mock_write.assert_called_once()
    mock_refresh.assert_called_once()
    mock_slack.assert_not_called()


@patch("src.app._notify_slack_error")
@patch(
    "src.app.run_dashboard_post_pipeline_checks",
    side_effect=RuntimeError("refresh down"),
)
@patch("src.app.write_predictions_to_database")
@patch("src.app.generate_win_predictions")
@patch("src.app.load_ml_model", return_value=MagicMock())
@patch("src.app.pull_tonights_games")
@patch("src.app.check_feature_flag", return_value=True)
@patch("src.app.get_feature_flags", return_value=pd.DataFrame())
@patch("src.app._create_db_engine")
def test_run_pipeline_returns_1_when_refresh_fails(
    mock_create_engine,
    _get_flags,
    _check_flag,
    mock_pull_games,
    _load_model,
    mock_generate,
    mock_write,
    _refresh,
    mock_slack,
    mock_engine,
    sample_games,
    sample_predictions,
):
    mock_create_engine.return_value = mock_engine
    mock_pull_games.return_value = sample_games
    mock_generate.return_value = sample_predictions

    assert app.run_pipeline() == 1
    mock_write.assert_called_once()
    mock_slack.assert_called_once()
    assert "ml_predictions" in mock_slack.call_args[0][0]


@patch("src.app._notify_slack_error")
@patch("src.app.pull_tonights_games", side_effect=RuntimeError("db error"))
@patch("src.app.check_feature_flag", return_value=True)
@patch("src.app.get_feature_flags", return_value=pd.DataFrame())
@patch("src.app._create_db_engine")
def test_run_pipeline_returns_1_on_unhandled_error(
    mock_create_engine,
    _get_flags,
    _check_flag,
    _pull_games,
    mock_slack,
    mock_engine,
):
    mock_create_engine.return_value = mock_engine

    assert app.run_pipeline() == 1
    mock_slack.assert_called_once()
    assert "NBA ELT ML Pipeline failed" in mock_slack.call_args[0][0]


@patch("src.app.write_to_slack")
def test_notify_slack_error_skips_when_webhook_unset(mock_write, monkeypatch):
    monkeypatch.delenv(app.SLACK_WEBHOOK_ENV, raising=False)
    logger = MagicMock()

    app._notify_slack_error("test message", logger)

    mock_write.assert_not_called()


@patch("src.app.write_to_slack")
def test_notify_slack_error_sends_when_webhook_set(mock_write, monkeypatch):
    monkeypatch.setenv(app.SLACK_WEBHOOK_ENV, "https://hooks.slack.com/test")
    logger = MagicMock()

    app._notify_slack_error("test message", logger)

    mock_write.assert_called_once_with(
        message="test message",
        webhook_url="https://hooks.slack.com/test",
    )
