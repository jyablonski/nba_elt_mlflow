from src.utils import check_feature_flag


def test_check_feature_flag_true(feature_flags_dataframe):
    feature_flag_boolean = check_feature_flag(
        flag="season", flags_df=feature_flags_dataframe
    )

    assert feature_flag_boolean is True


def test_check_feature_flag_false(feature_flags_dataframe):
    feature_flag_boolean = check_feature_flag(
        flag="playoffs", flags_df=feature_flags_dataframe
    )

    assert feature_flag_boolean is False
