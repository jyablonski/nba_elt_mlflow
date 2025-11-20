from jyablonski_common_modules.sql import write_to_sql_upsert
import pandas as pd

from src.utils import calculate_win_pct


def test_calculate_win_pct_postgres(postgres_conn, ml_model):
    ml_post_prediction_table = "ml_game_predictions"
    source_ml_schema = "silver"
    destination_ml_schema = "gold"
    count_check = f"SELECT count(*) FROM {source_ml_schema}.{ml_post_prediction_table}"
    count_check_results_before = pd.read_sql_query(sql=count_check, con=postgres_conn)

    tonights_games_full = pd.read_sql_query(
        "select * from ml_tonights_games", postgres_conn
    ).sort_values("home_team_avg_pts_scored")

    predictions = calculate_win_pct(full_df=tonights_games_full, ml_model=ml_model)

    write_to_sql_upsert(
        conn=postgres_conn,
        table=ml_post_prediction_table,
        schema=destination_ml_schema,
        df=predictions,
        primary_keys=["home_team", "game_date"],
    )

    count_check_results_after = pd.read_sql_query(sql=count_check, con=postgres_conn)

    assert len(predictions) == 4
    assert count_check_results_before["count"][0] == 0
    assert count_check_results_after["count"][0] == 4
