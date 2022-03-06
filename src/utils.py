from datetime import datetime, timedelta
import logging
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import exc, create_engine


def sql_connection(rds_schema: str):
    """
    SQL Connection function connecting to my postgres db with schema = nba_source where initial data in ELT lands
    Args:
        None
    Returns:
        SQL Connection variable to schema: nba_source in my PostgreSQL DB
    """
    RDS_USER = os.environ.get("RDS_USER")
    RDS_PW = os.environ.get("RDS_PW")
    RDS_IP = os.environ.get("IP")
    RDS_DB = os.environ.get("RDS_DB")
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
        df = pd.DataFrame(ml_model.predict_proba(ml_df)).rename(
            columns={0: "away_team_predicted_win_pct", 1: "home_team_predicted_win_pct"}
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

    except BaseException as e:
        logging.info(f"Error Occurred, {e}")
        df = []
        return df
