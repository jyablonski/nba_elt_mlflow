from unittest.mock import patch
from src.utils import load_v2_artifacts


def test_load_v2_artifacts_success():
    """Test that valid artifacts are returned when file exists."""
    expected_data = {"model": "fake_model", "feature_names": ["a", "b"]}

    # We patch the TrainingPipeline class *inside* src.utils
    with patch(
        "src.utils.TrainingPipeline.load_artifacts", return_value=expected_data
    ) as mock_load:
        result = load_v2_artifacts("dummy/path/model.joblib")

        assert result == expected_data
        mock_load.assert_called_once_with("dummy/path/model.joblib")


def test_load_v2_artifacts_file_not_found():
    """Test that an empty dict is returned (graceful failure) when file is missing."""

    # Force the loader to raise FileNotFoundError
    with patch(
        "src.utils.TrainingPipeline.load_artifacts", side_effect=FileNotFoundError
    ):
        result = load_v2_artifacts("non_existent_file.joblib")

        # It should catch the error and return {}
        assert result == {}
        assert isinstance(result, dict)
