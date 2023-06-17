from datetime import datetime
import logging
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import exc, create_engine
from sqlalchemy.engine.base import Connection, Engine


def sql_connection(
    rds_schema: str,
    RDS_USER: str = os.environ.get("RDS_USER"),
    RDS_PW: str = os.environ.get("RDS_PW"),
    RDS_IP: str = os.environ.get("IP"),
    RDS_DB: str = os.environ.get("RDS_DB"),
) -> Engine:
    """
    SQL Connection function to define the SQL Driver + connection variables needed to connect to the DB.
    This doesn't actually make the connection, use conn.connect() in a context manager to create 1 re-usable connection

    Args:
        rds_schema (str): The Schema in the DB to connect to.

    Returns:
        SQL Connection variable to a specified schema in my PostgreSQL DB
    """
    try:
        connection = create_engine(
            f"postgresql+psycopg2://{RDS_USER}:{RDS_PW}@{RDS_IP}:5432/{RDS_DB}",
            connect_args={"options": f"-csearch_path={rds_schema}"},
            # defining schema to connect to
            echo=False,
        )
        logging.info(f"SQL Connection to schema: {rds_schema} Successful")
        return connection
    except exc.SQLAlchemyError as e:
        logging.error(f"SQL Connection to schema: {rds_schema} Failed, Error: {e}")
        return e


def write_to_sql(con, table_name: str, df: pd.DataFrame, table_type: str):
    """
    SQL Table function to write a pandas data frame in aws_dfname_source format
    Args:
        data: The Pandas DataFrame to store in SQL
        table_type: Whether the table should replace or append to an existing SQL Table under that name
    Returns:
        Writes the Pandas DataFrame to a Table in Snowflake in the {nba_source} Schema we connected to.
    """
    try:
        if len(df) == 0:
            logging.info(f"{table_name} is empty, not writing to SQL")
        elif df.schema == "Validated":
            df.to_sql(
                con=con, name=f"{table_name}", index=False, if_exists=table_type,
            )
            logging.info(
                f"Writing {len(df)} {table_name} rows to aws_{table_name}_source to SQL"
            )
        else:
            logging.info(f"{table_name} Schema Invalidated, not writing to SQL")
    except BaseException as error:
        logging.error(f"SQL Write Script Failed, {error}")
        return error


def calculate_win_pct(
    ml_df: pd.DataFrame, full_df: pd.DataFrame, ml_model: LogisticRegression
) -> pd.DataFrame:
    try:
        if len(full_df) > 0:
            latest_date = pd.to_datetime(
                pd.to_datetime(full_df["proper_date"].drop_duplicates()).values[0]
            ).date()
            if latest_date != datetime.now().date():
                logging.error(
                    "Exiting out, date in tonights_games_ml isn't Today's Date"
                )
                df = []
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
            df = []
            return df

    except BaseException as e:
        logging.error(f"Error Occurred, {e}")
        df = []
        return df


def get_feature_flags(connection: Connection) -> pd.DataFrame:
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
