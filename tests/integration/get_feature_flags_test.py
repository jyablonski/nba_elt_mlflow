import pandas as pd

from src.utils import get_feature_flags


def test_get_feature_flags_postgres(postgres_conn):
    df = get_feature_flags(postgres_conn)

    assert len(df) == 2
    assert df['flag'][0] == 'season'
    assert df['is_enabled'][0] == 1

    assert df['flag'][1] == 'playoffs'
    assert df['is_enabled'][1] == 0