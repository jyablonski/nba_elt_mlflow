import pandas as pd

from src.utils import calculate_win_pct, write_to_sql


def test_calculate_win_pct_postgres(postgres_conn, ml_model):
    count_check = "SELECT count(*) FROM ml_models.tonights_games_ml"
    count_check_results_before = pd.read_sql_query(sql=count_check, con=postgres_conn)

    tonights_games_full = pd.read_sql_query(
        "select * from ml_tonights_games", postgres_conn
    ).sort_values("home_team_avg_pts_scored")

    predictions = calculate_win_pct(full_df=tonights_games_full, ml_model=ml_model)
    write_to_sql(postgres_conn, "tonights_games_ml", predictions, "append")

    count_check_results_after = pd.read_sql_query(sql=count_check, con=postgres_conn)

    assert len(predictions) == 4
    assert count_check_results_before["count"][0] == 0
    assert count_check_results_after["count"][0] == 4
