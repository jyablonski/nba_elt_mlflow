"""
NBA Game Prediction ML Experiments Package.

This package provides a production-grade workflow for NBA win prediction,
featuring Time-Series validation, Financial ROI simulation, and
strict prevention of data leakage.

Modules:
    config:              Schema definitions (V2), constants, and model hyperparameters.
    data_generator:      Synthetic data generation with latent variable logic.
    feature_engineering: Feature transformations, fatigue modeling, and selection.
    models:              Model factory, ensemble implementations, and persistence.
    evaluation:          Metrics (MCC, Brier), calibration analysis, and betting simulation.
    training_pipeline:   End-to-end orchestration with TimeSeriesSplit CV.
    model_comparison:    Benchmarking and production readiness checks.
    run_experiments:     CLI entry point.

Quick Start (CLI):
    # Run with synthetic data (End-to-end test)
    python -m ml_experiments.run_experiments --synthetic --samples 2000

    # Run with real data + Hyperparameter Tuning + Financial Check
    python -m ml_experiments.run_experiments --data data/nba_games_v2.csv --tune --financial

Example Usage (Python API):
    from ml_experiments import TrainingPipeline, ModelComparison, ModelFactory

    # 1. Initialize Pipeline (Enforces Time-Series Split)
    pipeline = TrainingPipeline(test_size=0.2, cv_splits=5)

    # 2. Load & Prep (Splits data temporally to prevent leakage)
    pipeline.load_and_prep_data("data/nba_games_v2.csv")

    # 3. Train & Tune
    result = pipeline.train_model("xgboost", hyperparameter_tuning=True)

    # 4. Check Results
    print(f"Test AUC: {result.test_metrics.auc}")

    # 5. Save for Production
    pipeline.save_artifacts(result, "models/prod_model.joblib")
"""

__version__ = "0.2.0"

# Explicit imports for better IDE Autocompletion
from ml_experiments.config import FEATURE_COLUMNS, TARGET_COLUMN
from ml_experiments.data_generator import SyntheticDataGenerator
from ml_experiments.feature_engineering import FeatureEngineer
from ml_experiments.models import ModelFactory
from ml_experiments.evaluation import ModelEvaluator, EvaluationMetrics
from ml_experiments.training_pipeline import TrainingPipeline, TrainingResult
from ml_experiments.model_comparison import ModelComparison, ComparisonResult

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "SyntheticDataGenerator",
    "FeatureEngineer",
    "ModelFactory",
    "ModelEvaluator",
    "EvaluationMetrics",
    "TrainingPipeline",
    "TrainingResult",
    "ModelComparison",
    "ComparisonResult",
]
