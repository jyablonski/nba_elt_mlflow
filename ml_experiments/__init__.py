"""
NBA Game Prediction ML Experiments Package.

This package provides improved training, data generation, model exploration,
and comparison utilities for NBA win prediction models.

Modules:
    config: Configuration and constants for features, models, and hyperparameters
    data_generator: Synthetic data generation for training augmentation
    feature_engineering: Feature transformations and selection methods
    models: Model factory and ensemble implementations
    evaluation: Comprehensive evaluation metrics and analysis
    training_pipeline: End-to-end training with CV and hyperparameter tuning
    model_comparison: Model benchmarking and production readiness assessment
    run_experiments: Main experiment runner script

Quick Start:
    # Run with synthetic data
    python -m ml_experiments.run_experiments --synthetic --samples 5000

    # Run with real data
    python -m ml_experiments.run_experiments --data path/to/data.csv --tune

    # Feature analysis
    python -m ml_experiments.run_experiments --data path/to/data.csv --analyze-features

Example Usage:
    from ml_experiments import TrainingPipeline, ModelComparison

    # Initialize pipeline
    pipeline = TrainingPipeline(random_state=42)

    # Load and prepare data
    df = pipeline.load_data("data.csv")
    X, y = pipeline.prepare_data(df)
    pipeline.split_data(X, y)

    # Train all available models
    results = pipeline.train_all_models(hyperparameter_tuning=True)

    # Compare models
    comparison = ModelComparison()
    result = comparison.compare_models(...)
"""

__version__ = "0.1.0"

# Lazy imports to avoid import errors if optional dependencies aren't installed
def __getattr__(name):
    """Lazy import of package components."""
    if name == "FEATURE_COLUMNS":
        from ml_experiments.config import FEATURE_COLUMNS
        return FEATURE_COLUMNS
    elif name == "TARGET_COLUMN":
        from ml_experiments.config import TARGET_COLUMN
        return TARGET_COLUMN
    elif name == "MODEL_CONFIGS":
        from ml_experiments.config import MODEL_CONFIGS
        return MODEL_CONFIGS
    elif name == "SyntheticDataGenerator":
        from ml_experiments.data_generator import SyntheticDataGenerator
        return SyntheticDataGenerator
    elif name == "FeatureEngineer":
        from ml_experiments.feature_engineering import FeatureEngineer
        return FeatureEngineer
    elif name == "ModelFactory":
        from ml_experiments.models import ModelFactory
        return ModelFactory
    elif name == "ModelEvaluator":
        from ml_experiments.evaluation import ModelEvaluator
        return ModelEvaluator
    elif name == "TrainingPipeline":
        from ml_experiments.training_pipeline import TrainingPipeline
        return TrainingPipeline
    elif name == "ModelComparison":
        from ml_experiments.model_comparison import ModelComparison
        return ModelComparison
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "MODEL_CONFIGS",
    "SyntheticDataGenerator",
    "FeatureEngineer",
    "ModelFactory",
    "ModelEvaluator",
    "TrainingPipeline",
    "ModelComparison",
]
