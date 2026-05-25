import os
import sys
from pathlib import Path

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

SOURCE_SCHEMA = "silver"
DESTINATION_SCHEMA = "gold"
SEASON_FEATURE_FLAG = "season"
MODEL_PATH = Path(__file__).resolve().parent / "log_model.joblib"
SLACK_WEBHOOK_ENV = "SLACK_WEBHOOK_URL"


def _create_db_engine():
    return create_sql_engine(
        user=os.environ.get("RDS_USER", "default"),
        password=os.environ.get("RDS_PW", "default"),
        host=os.environ.get("IP", "postgres"),
        database=os.environ.get("RDS_DB", "default"),
        schema=SOURCE_SCHEMA,
        port=int(os.environ.get("RDS_PORT", 5432)),
    )


def _notify_slack_error(message: str, logger) -> None:
    webhook_url = os.environ.get(SLACK_WEBHOOK_ENV, "").strip()
    if not webhook_url:
        return

    try:
        write_to_slack(message=message, webhook_url=webhook_url)
    except Exception:
        logger.exception("Failed to send Slack alert")


def run_pipeline() -> int:
    """
    Run the ML predictions pipeline.

    Returns 0 on success or intentional skip (no games, flag off, etc.).
    Returns 1 on operational failure (unhandled error or dashboard refresh failure).
    """
    logger = create_logger()
    logger.info("Starting NBA ELT ML Pipeline")

    engine = _create_db_engine()
    try:
        with engine.connect() as connection:
            with connection.begin():
                feature_flags = get_feature_flags(
                    connection=connection, schema=DESTINATION_SCHEMA
                )

                if not check_feature_flag(
                    flag=SEASON_FEATURE_FLAG, flags_df=feature_flags
                ):
                    logger.info("Season feature flag is disabled, exiting script")
                    return 0

                tonights_games = pull_tonights_games(
                    connection=connection, schema=SOURCE_SCHEMA
                )

                if tonights_games.empty:
                    logger.warning("No games found for prediction, exiting script")
                    return 0

        model = load_ml_model(model_path=str(MODEL_PATH))
        predictions = generate_win_predictions(games_df=tonights_games, ml_model=model)

        if predictions.empty:
            logger.warning("No predictions generated, exiting script")
            return 0

        with engine.begin() as connection:
            write_predictions_to_database(
                connection=connection,
                predictions_df=predictions,
                schema=DESTINATION_SCHEMA,
            )

        logger.info("NBA ELT ML predictions committed successfully")

        try:
            run_dashboard_post_pipeline_checks()
        except Exception as exc:
            logger.exception(
                "ML predictions were committed but dashboard post-checks failed"
            )
            _notify_slack_error(
                f"{format_slack_alert(exc)}\n\nPipeline step: ml_predictions",
                logger,
            )
            return 1

        logger.info("Finished NBA ELT ML Pipeline")
        return 0
    except Exception as exc:
        logger.exception("NBA ELT ML Pipeline failed")
        _notify_slack_error(f"NBA ELT ML Pipeline failed: {exc}", logger)
        return 1
    finally:
        engine.dispose()


def main() -> None:
    sys.exit(run_pipeline())


if __name__ == "__main__":
    main()
