import pandas as pd

from src.utils import check_feature_flag, get_feature_flags


def test_get_feature_flags_postgres(postgres_conn):
    df = get_feature_flags(postgres_conn)

    assert len(df) == 2
    assert df["flag"][0] == "season"
    assert df["is_enabled"][0] == 1

    assert df["flag"][1] == "playoffs"
    assert df["is_enabled"][1] == 0


def test_get_and_check_feature_flags_postgres(postgres_conn):
    df = get_feature_flags(postgres_conn)

    season_check = check_feature_flag(flag="season", flags_df=df)
    playoffs_check = check_feature_flag(flag="playoffs", flags_df=df)

    assert season_check == True
    assert playoffs_check == False
