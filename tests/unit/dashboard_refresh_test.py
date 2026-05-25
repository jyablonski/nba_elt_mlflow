from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.dashboard_refresh import (
    DASHBOARD_HEALTH_URL,
    DashboardCheckError,
    format_slack_alert,
    run_dashboard_post_pipeline_checks,
    validate_health,
    validate_refresh,
)


@pytest.fixture(autouse=True)
def clear_refresh_env(monkeypatch):
    monkeypatch.delenv("DASHBOARD_REFRESH_URL", raising=False)
    monkeypatch.delenv("DATA_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("DASHBOARD_REFRESH_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DASHBOARD_HEALTH_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DASHBOARD_REFRESH_MAX_ATTEMPTS", raising=False)


def _refresh_payload(**overrides):
    base = {
        "status": "ok",
        "duration_seconds": 0.08,
        "row_counts": {
            "bans": 2,
            "standings": 30,
            "player_stats": 562,
            "team_ratings": 30,
            "pbp": 1843,
        },
    }
    base.update(overrides)
    return base


def _health_payload(**overrides):
    base = {
        "status": "ok",
        "has_snapshot": True,
        "last_refreshed_at": datetime.now(timezone.utc).isoformat(),
        "memory": {"container_current_mb": 208.2, "process_rss_mb": 141.5},
        "row_counts": {
            "bans": 2,
            "standings": 30,
            "player_stats": 562,
            "team_ratings": 30,
            "pbp": 1843,
        },
    }
    base.update(overrides)
    return base


def test_skipped_when_refresh_url_unset(monkeypatch):
    monkeypatch.setenv("DASHBOARD_REFRESH_URL", "")

    with patch("src.dashboard_refresh.requests.request") as request:
        run_dashboard_post_pipeline_checks()

    request.assert_not_called()


def test_raises_when_token_missing(monkeypatch):
    monkeypatch.setenv(
        "DASHBOARD_REFRESH_URL", "https://example.com/internal/refresh-data"
    )

    with pytest.raises(DashboardCheckError, match="DATA_REFRESH_TOKEN"):
        run_dashboard_post_pipeline_checks()


def test_success_runs_refresh_then_health(monkeypatch):
    monkeypatch.setenv(
        "DASHBOARD_REFRESH_URL", "https://example.com/internal/refresh-data"
    )
    monkeypatch.setenv("DATA_REFRESH_TOKEN", "secret-token")
    monkeypatch.setenv("DASHBOARD_REFRESH_MAX_ATTEMPTS", "1")

    refresh_response = MagicMock(status_code=200)
    refresh_response.json.return_value = _refresh_payload()
    health_response = MagicMock(status_code=200)
    health_response.json.side_effect = lambda: _health_payload()

    with patch(
        "src.dashboard_refresh.requests.request",
        side_effect=[refresh_response, health_response],
    ) as request:
        run_dashboard_post_pipeline_checks()

    assert request.call_count == 2
    assert request.call_args_list[0][0][0] == "POST"
    assert request.call_args_list[1][0][0] == "GET"
    assert request.call_args_list[1][0][1] == DASHBOARD_HEALTH_URL


def test_refresh_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv(
        "DASHBOARD_REFRESH_URL", "https://example.com/internal/refresh-data"
    )
    monkeypatch.setenv("DATA_REFRESH_TOKEN", "secret-token")
    monkeypatch.setenv("DASHBOARD_REFRESH_MAX_ATTEMPTS", "3")

    refresh_response = MagicMock(status_code=200)
    refresh_response.json.return_value = _refresh_payload()
    health_response = MagicMock(status_code=200)
    health_response.json.side_effect = lambda: _health_payload()

    with (
        patch("src.dashboard_refresh.time.sleep"),
        patch(
            "src.dashboard_refresh.requests.request",
            side_effect=[
                requests.Timeout(),
                refresh_response,
                health_response,
            ],
        ) as request,
    ):
        run_dashboard_post_pipeline_checks()

    assert request.call_count == 3


def test_refresh_rejects_zero_critical_table():
    row_counts = _refresh_payload()["row_counts"]
    row_counts["bans"] = 0

    with pytest.raises(DashboardCheckError, match="zero_or_missing_tables"):
        validate_refresh(_refresh_payload(row_counts=row_counts))


def test_health_rejects_stale_snapshot_timestamp():
    started = datetime(2026, 5, 25, 18, 44, 20, tzinfo=timezone.utc)
    payload = _health_payload(last_refreshed_at="2026-05-25T18:44:14.328520+00:00")

    with pytest.raises(DashboardCheckError, match="stale"):
        validate_health(payload, started)


def test_format_slack_alert_includes_context():
    exc = DashboardCheckError(
        "NBA dashboard health check failed",
        endpoint="/internal/health",
        status=503,
    )
    message = format_slack_alert(exc)

    assert "NBA dashboard health check failed" in message
    assert "/internal/health" in message
    assert "503" in message


def test_memory_warning_only_logs(caplog, monkeypatch):
    monkeypatch.setenv(
        "DASHBOARD_REFRESH_URL", "https://example.com/internal/refresh-data"
    )
    monkeypatch.setenv("DATA_REFRESH_TOKEN", "secret-token")
    monkeypatch.setenv("DASHBOARD_REFRESH_MAX_ATTEMPTS", "1")

    refresh_response = MagicMock(status_code=200)
    refresh_response.json.return_value = _refresh_payload()
    health_response = MagicMock(status_code=200)
    health_response.json.side_effect = lambda: _health_payload(
        memory={"container_current_mb": 900, "process_rss_mb": 141.5}
    )

    with patch(
        "src.dashboard_refresh.requests.request",
        side_effect=[refresh_response, health_response],
    ):
        run_dashboard_post_pipeline_checks()

    assert "Dashboard memory warning" in caplog.text
