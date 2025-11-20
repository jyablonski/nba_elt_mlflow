import pytest

from src.utils import check_feature_flag, get_feature_flags


def test_get_feature_flags_postgres(postgres_conn):
    """Test retrieving feature flags from database."""
    flags = get_feature_flags(connection=postgres_conn, schema="gold")

    assert len(flags) == 2, "Should have 2 feature flags"

    # Check season flag
    season_flag = flags[flags["flag"] == "season"].iloc[0]
    assert season_flag["is_enabled"] == 1, "Season flag should be enabled"

    # Check playoffs flag
    playoffs_flag = flags[flags["flag"] == "playoffs"].iloc[0]
    assert playoffs_flag["is_enabled"] == 0, "Playoffs flag should be disabled"


def test_check_feature_flag_enabled(feature_flags_dataframe):
    """Test checking an enabled feature flag."""
    result = check_feature_flag(flag="season", flags_df=feature_flags_dataframe)
    assert result is True, "Season flag should be enabled"


def test_check_feature_flag_disabled(feature_flags_dataframe):
    """Test checking a disabled feature flag."""
    result = check_feature_flag(flag="playoffs", flags_df=feature_flags_dataframe)
    assert result is False, "Playoffs flag should be disabled"


def test_check_feature_flag_nonexistent(feature_flags_dataframe):
    """Test checking a feature flag that doesn't exist."""
    result = check_feature_flag(flag="nonexistent", flags_df=feature_flags_dataframe)
    assert result is False, "Nonexistent flag should return False"


def test_get_and_check_feature_flags_integration(postgres_conn):
    """Integration test for getting and checking feature flags."""
    flags = get_feature_flags(connection=postgres_conn, schema="gold")

    season_enabled = check_feature_flag(flag="season", flags_df=flags)
    playoffs_enabled = check_feature_flag(flag="playoffs", flags_df=flags)

    assert season_enabled is True, "Season should be enabled"
    assert playoffs_enabled is False, "Playoffs should be disabled"


@pytest.mark.parametrize(
    "flag_name,expected",
    [
        ("season", True),
        ("playoffs", False),
    ],
)
def test_feature_flags_parametrized(feature_flags_dataframe, flag_name, expected):
    """Parametrized test for multiple feature flags."""
    result = check_feature_flag(flag=flag_name, flags_df=feature_flags_dataframe)
    assert result == expected, f"Flag '{flag_name}' should be {expected}"
