import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

DASHBOARD_HEALTH_URL = "https://nbadashboard.jyablonski.dev/internal/health"

CRITICAL_TABLES = [
    "bans",
    "standings",
    "player_stats",
    "team_ratings",
    "pbp",
]

MEMORY_WARNINGS_MB = {
    "container_current_mb": 750,
    "process_rss_mb": 500,
}

DEFAULT_REFRESH_TIMEOUT_SECONDS = 30
DEFAULT_HEALTH_TIMEOUT_SECONDS = 15
DEFAULT_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2


class DashboardCheckError(RuntimeError):
    """Operational failure during dashboard refresh or health validation."""

    def __init__(self, title: str, **context: Any):
        self.title = title
        self.context = context
        detail = ", ".join(f"{key}={value}" for key, value in context.items())
        super().__init__(f"{title}" + (f" ({detail})" if detail else ""))


def format_slack_alert(exc: BaseException) -> str:
    """Build a compact Slack message from a dashboard check failure."""
    if isinstance(exc, DashboardCheckError):
        lines = [exc.title]
        for key, value in exc.context.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    return f"NBA dashboard post-pipeline check failed\n\nError: {exc}"


def run_dashboard_post_pipeline_checks() -> None:
    """
    Refresh dashboard memory, then verify health reflects a fresh snapshot.

    Skips entirely when DASHBOARD_REFRESH_URL is unset (local runs).
    """
    refresh_url = os.environ.get("DASHBOARD_REFRESH_URL", "").strip()
    if not refresh_url:
        logger.info("DASHBOARD_REFRESH_URL not set; skipping dashboard refresh")
        return

    token = _require_refresh_token()
    refresh_started_at = datetime.now(timezone.utc)

    refresh_payload = _post_refresh(
        url=refresh_url,
        token=token,
        timeout=_refresh_timeout(),
        max_attempts=_max_attempts(),
    )
    validate_refresh(refresh_payload)
    _log_refresh_success(refresh_payload)

    health_payload = _get_health(
        token=token,
        timeout=_health_timeout(),
        max_attempts=_max_attempts(),
    )
    validate_health(health_payload, refresh_started_at)
    _log_health_success(health_payload)
    _warn_memory_thresholds(health_payload.get("memory") or {})


def _require_refresh_token() -> str:
    token = os.environ.get("DATA_REFRESH_TOKEN", "").strip()
    if not token:
        raise DashboardCheckError(
            "NBA dashboard refresh failed",
            endpoint="/internal/refresh-data",
            error="DATA_REFRESH_TOKEN is required when DASHBOARD_REFRESH_URL is set",
        )
    return token


def _refresh_timeout() -> int:
    return int(
        os.environ.get(
            "DASHBOARD_REFRESH_TIMEOUT_SECONDS", DEFAULT_REFRESH_TIMEOUT_SECONDS
        )
    )


def _health_timeout() -> int:
    return int(
        os.environ.get("DASHBOARD_HEALTH_TIMEOUT_SECONDS", DEFAULT_HEALTH_TIMEOUT_SECONDS)
    )


def _max_attempts() -> int:
    return int(os.environ.get("DASHBOARD_REFRESH_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))


def _request_json(
    *,
    method: str,
    url: str,
    token: str,
    timeout: int,
    max_attempts: int,
    endpoint_label: str,
) -> dict[str, Any]:
    last_error: BaseException | None = None
    headers = {"X-Refresh-Token": token}

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code != 200:
                raise DashboardCheckError(
                    f"NBA dashboard {endpoint_label} failed",
                    endpoint=endpoint_label,
                    status=response.status_code,
                    body=response.text[:500],
                )
            payload = response.json()
            if not isinstance(payload, dict):
                raise DashboardCheckError(
                    f"NBA dashboard {endpoint_label} failed",
                    endpoint=endpoint_label,
                    error="response JSON was not an object",
                )
            return payload
        except DashboardCheckError:
            raise
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "Dashboard %s attempt %s/%s failed: %s",
                endpoint_label,
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise DashboardCheckError(
        f"NBA dashboard {endpoint_label} failed",
        endpoint=endpoint_label,
        error=str(last_error),
        attempts=max_attempts,
    ) from last_error


def _post_refresh(url: str, token: str, timeout: int, max_attempts: int) -> dict[str, Any]:
    return _request_json(
        method="POST",
        url=url,
        token=token,
        timeout=timeout,
        max_attempts=max_attempts,
        endpoint_label="/internal/refresh-data",
    )


def _get_health(token: str, timeout: int, max_attempts: int) -> dict[str, Any]:
    return _request_json(
        method="GET",
        url=DASHBOARD_HEALTH_URL,
        token=token,
        timeout=timeout,
        max_attempts=max_attempts,
        endpoint_label="/internal/health",
    )


def validate_refresh(payload: dict[str, Any]) -> None:
    if payload.get("status") != "ok":
        raise DashboardCheckError(
            "NBA dashboard refresh failed",
            endpoint="/internal/refresh-data",
            status_payload=payload.get("status"),
        )

    row_counts = payload.get("row_counts")
    if row_counts is None and payload.get("tables") is not None:
        tables = payload["tables"]
        row_counts = (
            {name: 1 for name in tables}
            if isinstance(tables, list)
            else tables
        )

    if not isinstance(row_counts, dict):
        raise DashboardCheckError(
            "NBA dashboard refresh failed",
            endpoint="/internal/refresh-data",
            error="row_counts missing from refresh response",
        )

    _validate_critical_row_counts(row_counts, endpoint="/internal/refresh-data")


def validate_health(payload: dict[str, Any], refresh_started_at: datetime) -> None:
    if payload.get("status") != "ok":
        raise DashboardCheckError(
            "NBA dashboard health check failed",
            endpoint="/internal/health",
            status_payload=payload.get("status"),
            has_snapshot=payload.get("has_snapshot"),
        )

    if payload.get("has_snapshot") is not True:
        raise DashboardCheckError(
            "NBA dashboard health check failed",
            endpoint="/internal/health",
            has_snapshot=payload.get("has_snapshot"),
            error="expected has_snapshot=true",
        )

    last_refreshed_raw = payload.get("last_refreshed_at")
    if not last_refreshed_raw:
        raise DashboardCheckError(
            "NBA dashboard health check failed",
            endpoint="/internal/health",
            last_refreshed_at=last_refreshed_raw,
            error="last_refreshed_at missing",
        )

    last_refreshed_at = _parse_utc_timestamp(str(last_refreshed_raw))
    if last_refreshed_at < refresh_started_at:
        raise DashboardCheckError(
            "NBA dashboard health check failed",
            endpoint="/internal/health",
            last_refreshed_at=last_refreshed_raw,
            refresh_started_at=refresh_started_at.isoformat(),
            error="snapshot timestamp is stale",
        )

    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, dict):
        raise DashboardCheckError(
            "NBA dashboard health check failed",
            endpoint="/internal/health",
            error="row_counts missing from health response",
        )

    _validate_critical_row_counts(row_counts, endpoint="/internal/health")


def _validate_critical_row_counts(row_counts: dict[str, Any], *, endpoint: str) -> None:
    missing_or_zero = [
        table
        for table in CRITICAL_TABLES
        if not row_counts.get(table)
    ]
    if missing_or_zero:
        raise DashboardCheckError(
            "NBA dashboard validation failed",
            endpoint=endpoint,
            zero_or_missing_tables=",".join(missing_or_zero),
            row_counts={table: row_counts.get(table) for table in CRITICAL_TABLES},
        )


def _parse_utc_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _warn_memory_thresholds(memory: dict[str, Any]) -> None:
    for field, threshold in MEMORY_WARNINGS_MB.items():
        current = memory.get(field)
        if current is None:
            continue
        if float(current) > threshold:
            logger.warning(
                "Dashboard memory warning: %s=%s MB exceeds threshold %s MB",
                field,
                current,
                threshold,
            )


def _log_refresh_success(payload: dict[str, Any]) -> None:
    row_counts = payload.get("row_counts") or {}
    logger.info(
        "Dashboard refresh succeeded: duration=%ss row_count_keys=%s",
        payload.get("duration_seconds"),
        len(row_counts) if isinstance(row_counts, dict) else row_counts,
    )


def _log_health_success(payload: dict[str, Any]) -> None:
    memory = payload.get("memory") or {}
    row_counts = payload.get("row_counts") or {}
    logger.info(
        "Dashboard health ok: last_refreshed_at=%s container_mb=%s process_rss_mb=%s "
        "critical_tables=%s",
        payload.get("last_refreshed_at"),
        memory.get("container_current_mb"),
        memory.get("process_rss_mb"),
        {table: row_counts.get(table) for table in CRITICAL_TABLES},
    )
