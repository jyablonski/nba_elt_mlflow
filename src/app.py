import os
import sys
from enum import IntEnum
from pathlib import Path

from sqlalchemy.engine import Engine

from jyablonski_common_modules.general import write_to_slack
from jyablonski_common_modules.logging import create_logger
from jyablonski_common_modules.sql import create_sql_engine

from src.dashboard_refresh import (
    format_slack_alert,
    run_dashboard_post_pipeline_checks,
)
from src.utils import (
    check_feature_flag,
    generate_win_predictions,
    get_feature_flags,
    load_ml_model,
    pull_tonights_games,
    write_predictions_to_database,
)

logger = create_logger()

SOURCE_SCHEMA = "silver"
DESTINATION_SCHEMA = "gold"
SEASON_FEATURE_FLAG = "season"
MODEL_PATH = Path(__file__).resolve().parent / "log_model.joblib"
SLACK_WEBHOOK_ENV = "SLACK_WEBHOOK_URL"

# Labels describing how the prediction workflow ended. Used as the dashboard
# refresh / Slack alert context, and asserted on in tests.
STEP_SEASON_FLAG_DISABLED = "season_feature_flag_disabled"
STEP_NO_GAMES = "no_games_for_prediction"
STEP_NO_PREDICTIONS = "no_predictions_generated"
STEP_ML_PREDICTIONS = "ml_predictions"


class ExitCode(IntEnum):
    """Process exit codes returned by the pipeline."""

    SUCCESS = 0  # success or intentional skip (no games, flag off, etc.)
    FAILURE = 1  # operational failure (unhandled error or dashboard refresh failure)


def _notify_slack_error(message: str) -> None:
    webhook_url = os.environ.get(SLACK_WEBHOOK_ENV, "").strip()
    if not webhook_url:
        return

    try:
        write_to_slack(message=message, webhook_url=webhook_url)
    except Exception:
        logger.exception("Failed to send Slack alert")


def _refresh_dashboard_and_alert(pipeline_step: str) -> ExitCode:
    """Refresh the dashboard, alerting Slack and returning FAILURE on error."""
    try:
        run_dashboard_post_pipeline_checks()
    except Exception as exc:
        logger.exception("Dashboard post-checks failed")
        _notify_slack_error(
            f"{format_slack_alert(exc)}\n\nPipeline step: {pipeline_step}"
        )
        return ExitCode.FAILURE

    return ExitCode.SUCCESS


def _generate_and_store_predictions(engine: Engine) -> str:
    """
    Run the prediction workflow and return the step label describing its outcome.

    The label is used downstream as dashboard-refresh / alerting context. Raises
    on operational failure; the caller is responsible for turning that into an
    exit code and Slack alert.
    """
    with engine.connect() as connection:
        with connection.begin():
            feature_flags = get_feature_flags(
                connection=connection, schema=DESTINATION_SCHEMA
            )

            if not check_feature_flag(
                flag=SEASON_FEATURE_FLAG, flags_df=feature_flags
            ):
                logger.info("Season feature flag is disabled, exiting script")
                return STEP_SEASON_FLAG_DISABLED

            tonights_games = pull_tonights_games(
                connection=connection, schema=SOURCE_SCHEMA
            )

    if tonights_games.empty:
        logger.warning("No games found for prediction, exiting script")
        return STEP_NO_GAMES

    model = load_ml_model(model_path=str(MODEL_PATH))
    predictions = generate_win_predictions(games_df=tonights_games, ml_model=model)

    if predictions.empty:
        logger.warning("No predictions generated, exiting script")
        return STEP_NO_PREDICTIONS

    with engine.begin() as connection:
        write_predictions_to_database(
            connection=connection,
            predictions_df=predictions,
            schema=DESTINATION_SCHEMA,
        )

    logger.info("NBA ELT ML predictions committed successfully")
    return STEP_ML_PREDICTIONS


def run_pipeline() -> ExitCode:
    """
    Run the ML predictions pipeline.

    Every non-error path ends with exactly one dashboard refresh, so the refresh
    is a single tail call. Returns ExitCode.SUCCESS on success or intentional skip
    (no games, flag off, etc.) and ExitCode.FAILURE on operational failure.
    """
    logger.info("Starting NBA ELT ML Pipeline")

    engine = create_sql_engine(
        user=os.environ.get("RDS_USER", "default"),
        password=os.environ.get("RDS_PW", "default"),
        host=os.environ.get("IP", "postgres"),
        database=os.environ.get("RDS_DB", "default"),
        schema=SOURCE_SCHEMA,
        port=int(os.environ.get("RDS_PORT", 5432)),
    )
    try:
        pipeline_step = _generate_and_store_predictions(engine)
        exit_code = _refresh_dashboard_and_alert(pipeline_step)
        if exit_code is ExitCode.SUCCESS:
            logger.info("Finished NBA ELT ML Pipeline")
        return exit_code
    except Exception as exc:
        logger.exception("NBA ELT ML Pipeline failed")
        _notify_slack_error(f"NBA ELT ML Pipeline failed: {exc}")
        return ExitCode.FAILURE
    finally:
        engine.dispose()


def main() -> None:
    sys.exit(run_pipeline())


if __name__ == "__main__":
    main()
