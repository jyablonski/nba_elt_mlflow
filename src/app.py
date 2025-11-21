import os
import sys

from jyablonski_common_modules.logging import create_logger
from jyablonski_common_modules.sql import create_sql_engine

from src.utils import (
    check_feature_flag,
    get_feature_flags,
    load_ml_model,
    pull_tonights_games,
    generate_win_predictions,
    write_predictions_to_database,
)

SOURCE_SCHEMA = "silver"
DESTINATION_SCHEMA = "gold"
MODEL_PATH = "src/log_model.joblib"


def main():
    """Main execution function for NBA ML predictions pipeline."""
    logger = create_logger()
    logger.info("Starting NBA ELT ML Pipeline")

    engine = create_sql_engine(
        user=os.environ.get("RDS_USER", "default"),
        password=os.environ.get("RDS_PW", "default"),
        host=os.environ.get("IP", "postgres"),
        database=os.environ.get("RDS_DB", "default"),
        schema=SOURCE_SCHEMA,
        port=int(os.environ.get("RDS_PORT", 5432)),
    )

    with engine.begin() as connection:
        feature_flags = get_feature_flags(
            connection=connection, schema=DESTINATION_SCHEMA
        )

        if not check_feature_flag(flag="season", flags_df=feature_flags):
            logger.info("Season feature flag is disabled, exiting script")
            sys.exit(0)

        tonights_games = pull_tonights_games(
            connection=connection, schema=SOURCE_SCHEMA
        )

        if tonights_games.empty:
            logger.warning("No games found for prediction, exiting script")
            sys.exit(0)

        model = load_ml_model(model_path=MODEL_PATH)

        predictions = generate_win_predictions(games_df=tonights_games, ml_model=model)

        if predictions.empty:
            logger.warning("No predictions generated, exiting script")
            sys.exit(0)

        write_predictions_to_database(
            connection=connection,
            predictions_df=predictions,
            schema=DESTINATION_SCHEMA,
        )

    logger.info("Finished NBA ELT ML Pipeline")


if __name__ == "__main__":
    main()
