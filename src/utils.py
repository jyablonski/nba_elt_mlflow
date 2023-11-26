from datetime import datetime
import logging
import os
from typing import Literal

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import exc, create_engine
from sqlalchemy.engine.base import Connection, Engine


def sql_connection(
    rds_schema: str,
    rds_user: str = os.environ.get("RDS_USER", "postgres"),
    rds_pw: str = os.environ.get("RDS_PW", "postgres"),
    rds_ip: str = os.environ.get("IP", "postgres"),
    rds_db: str = os.environ.get("RDS_DB", "postgres"),
) -> Engine:
    """
    SQL Connection function to define the SQL Driver + connection
    variables needed to connect to the DB.

    This doesn't actually make the connection, use conn.connect()
    in a context manager to create 1 re-usable connection

    Args:
        rds_schema (str): The Schema in the DB to connect to.

    Returns:
        SQL Connection variable to a specified schema in my PostgreSQL DB
    """
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{rds_user}:{rds_pw}@{rds_ip}:5432/{rds_db}",
            # pool_size=0,
            # max_overflow=20,
            connect_args={
                "options": f"-csearch_path={rds_schema}",
            },
            # defining schema to connect to
            echo=False,
        )
        logging.info(f"SQL Engine for {rds_ip}:5432/{rds_db}/{rds_schema} created")
        return engine
    except exc.SQLAlchemyError as e:
        logging.error(f"SQL Engine for {rds_ip}:5432/{rds_db}/{rds_schema} failed, {e}")
        raise e


def write_to_sql(
    con,
    table_name: str,
    df: pd.DataFrame,
    table_type: Literal["fail", "replace", "append"] = "append",
) -> None:
    """
    SQL Table function to write a pandas data frame in aws_dfname_source format

    Args:
        con (SQL Connection): The connection to the SQL DB.

        table_name (str): The Table name to write to SQL as.

        df (DataFrame): The Pandas DataFrame to store in SQL

        table_type (str): Whether the table should replace or append to an
            existing SQL Table under that name

    Returns:
        Writes the Pandas DataFrame to a Table in the Schema we connected to.

    """
    try:
        if len(df) == 0:
            logging.info(f"{table_name} is empty, not writing to SQL")
        else:
            df.to_sql(
                con=con,
                name=table_name,
                index=False,
                if_exists=table_type,
            )
            logging.info(f"Writing {len(df)} {table_name} rows to {table_name} to SQL")
    except BaseException as error:
        logging.error(f"SQL Write Script Failed, {error}")
        raise error


def calculate_win_pct(
    full_df: pd.DataFrame, ml_model: LogisticRegression
) -> pd.DataFrame:
    """
    Function to Calculate Win Prediction %s for Upcoming Games

    Args:
        full_df (pd.DataFrame): The Full DataFrame with other variables not
            needed for the ML Predictions

        ml_model (LogisticRegression): The ML Model loaded from
            `log_model.joblib`

    Returns:
        Pandas DataFrame of ML Predictions to append into `tonights_games_ml`
    """
    try:
        ml_df = full_df.drop(
            [
                "home_team",
                "home_moneyline",
                "away_team",
                "away_moneyline",
                "game_date",
                "outcome",
            ],
            axis=1,
        )

        if len(full_df) > 0:
            latest_date = pd.to_datetime(
                pd.to_datetime(full_df["game_date"].drop_duplicates()).values[0]
            ).date()
            if latest_date != datetime.utcnow().date():
                logging.error(
                    f"Exiting out, {latest_date} != {datetime.utcnow().date()}"
                )
                df = pd.DataFrame()
                return df

            else:
                df = pd.DataFrame(ml_model.predict_proba(ml_df)).rename(
                    columns={
                        0: "away_team_predicted_win_pct",
                        1: "home_team_predicted_win_pct",
                    }
                )
                df_final = full_df.reset_index().drop(
                    "outcome", axis=1
                )  # reset index so predictions match up correctly

                df_final["home_team_predicted_win_pct"] = df[
                    "home_team_predicted_win_pct"
                ].round(3)
                df_final["away_team_predicted_win_pct"] = df[
                    "away_team_predicted_win_pct"
                ].round(3)

                logging.info(f"Predicted Win %s for {len(df_final)} games")
                df_final.schema = "Validated"
                return df_final
        else:
            logging.error("Exiting out, don't have data for Today's Games")
            df = pd.DataFrame()
            return df

    except BaseException as e:
        logging.error(f"Error Occurred, {e}")
        df = pd.DataFrame()
        return df


def get_feature_flags(connection: Connection | Engine) -> pd.DataFrame:
    flags = pd.read_sql_query(
        sql="select * from nba_prod.feature_flags;", con=connection
    )

    print(f"Retrieving {len(flags)} Feature Flags")
    return flags


def check_feature_flag(flag: str, flags_df: pd.DataFrame) -> bool:
    flags_df = flags_df.query(f"flag == '{flag}'")

    if len(flags_df) > 0 and flags_df["is_enabled"].iloc[0] == 1:
        print(f"Feature Flag for {flag} is enabled, continuing")
        return True
    else:
        print(f"Feature Flag for {flag} is disabled, skipping")
        logging.info(f"Feature Flag for {flag} is disabled, skipping")
        return False
