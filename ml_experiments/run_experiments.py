"""
Main experiment runner for NBA game predictions.

This script demonstrates the full ML experimentation workflow including:
- Data loading and preprocessing
- Synthetic data generation
- Feature engineering
- Model training and hyperparameter tuning
- Model comparison and evaluation
- Production readiness assessment

Usage:
    python -m ml_experiments.run_experiments --data path/to/data.csv
    python -m ml_experiments.run_experiments --synthetic --samples 5000
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_experiments.config import FEATURE_COLUMNS, TARGET_COLUMN
from ml_experiments.data_generator import SyntheticDataGenerator
from ml_experiments.feature_engineering import FeatureEngineer
from ml_experiments.models import ModelFactory, create_recommended_models
from ml_experiments.evaluation import ModelEvaluator, calculate_baseline_accuracy
from ml_experiments.training_pipeline import TrainingPipeline
from ml_experiments.model_comparison import ModelComparison


def run_synthetic_experiment(
    n_samples: int = 5000,
    output_dir: Optional[str] = None,
    tune_hyperparameters: bool = False,
) -> None:
    """
    Run experiments using synthetic data.

    Useful for testing the pipeline and establishing baseline
    performance expectations.
    """
    print("=" * 60)
    print("SYNTHETIC DATA EXPERIMENT")
    print("=" * 60)

    # Generate synthetic data
    print(f"\nGenerating {n_samples} synthetic samples...")
    generator = SyntheticDataGenerator(random_state=42)
    df = generator.generate_samples(n_samples=n_samples)

    print(f"Generated dataset shape: {df.shape}")
    print(f"Class distribution:\n{df[TARGET_COLUMN].value_counts()}")

    # Run the pipeline
    _run_pipeline(df, output_dir, tune_hyperparameters)


def run_file_experiment(
    data_filepath: str,
    output_dir: Optional[str] = None,
    tune_hyperparameters: bool = False,
    augment_data: bool = False,
) -> None:
    """
    Run experiments using data from a file.

    Args:
        data_filepath: Path to CSV file with training data
        output_dir: Directory for output files
        tune_hyperparameters: Whether to tune hyperparameters
        augment_data: Whether to augment with synthetic data
    """
    print("=" * 60)
    print("FILE DATA EXPERIMENT")
    print("=" * 60)

    # Load data
    pipeline = TrainingPipeline()
    df = pipeline.load_data(data_filepath)

    print(f"Loaded dataset shape: {df.shape}")

    if TARGET_COLUMN in df.columns:
        print(f"Class distribution:\n{df[TARGET_COLUMN].value_counts()}")

    # Optionally augment data
    if augment_data:
        print("\nAugmenting data with synthetic samples...")
        generator = SyntheticDataGenerator(random_state=42)
        df = generator.augment_dataset(df, augmentation_ratio=0.3)
        print(f"Augmented dataset shape: {df.shape}")

    # Run the pipeline
    _run_pipeline(df, output_dir, tune_hyperparameters)


def _run_pipeline(
    df: pd.DataFrame,
    output_dir: Optional[str],
    tune_hyperparameters: bool,
) -> None:
    """Run the core ML pipeline on prepared data."""

    # Initialize components
    pipeline = TrainingPipeline(random_state=42, test_size=0.2, cv_folds=5)
    comparison = ModelComparison(random_state=42)
    factory = ModelFactory()

    # Prepare features
    print("\nEngineering features...")
    X, y = pipeline.prepare_data(df, add_derived_features=True)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    # Split data
    pipeline.split_data(X, y)
    print(f"\nTrain size: {len(pipeline.X_train)}")
    print(f"Test size: {len(pipeline.X_test)}")

    # Calculate baseline
    baseline_acc = calculate_baseline_accuracy(pipeline.y_test)
    print(f"Baseline accuracy (majority class): {baseline_acc:.4f}")

    # Get available models
    available_models = factory.get_installed_models()
    print(f"\nAvailable models: {available_models}")

    # Create models for comparison
    models = create_recommended_models(factory)

    # Run comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    result = comparison.compare_models(
        models=models,
        X_train=pipeline.X_train,
        y_train=pipeline.y_train,
        X_test=pipeline.X_test,
        y_test=pipeline.y_test,
        cv_folds=5,
        include_baseline=True,
    )

    # Print report
    report = comparison.generate_comparison_report(result)
    print("\n" + report)

    # Hyperparameter tuning for top models if requested
    if tune_hyperparameters:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)

        # Get top 3 models for tuning
        top_models = result.rankings.head(3)["model"].tolist()
        top_models = [m for m in top_models if m != "baseline"]

        tuned_results = {}
        for model_name in top_models:
            if model_name in factory.get_installed_models():
                print(f"\nTuning {model_name}...")
                tuned_result = pipeline.train_model(
                    model_name,
                    hyperparameter_tuning=True,
                    tuning_method="random",
                    n_iter=30,
                )
                tuned_results[model_name] = tuned_result
                print(f"Best params: {tuned_result.best_params}")
                print(f"Test ROC AUC: {tuned_result.test_metrics.roc_auc:.4f}")

    # Create ensemble from top performers
    print("\n" + "=" * 60)
    print("ENSEMBLE MODEL")
    print("=" * 60)

    # Get top 3 non-baseline models
    top_for_ensemble = [
        m for m in result.rankings.head(4)["model"].tolist()
        if m != "baseline" and m in factory.get_installed_models()
    ][:3]

    if len(top_for_ensemble) >= 2:
        print(f"Creating voting ensemble from: {top_for_ensemble}")
        ensemble_result = pipeline.create_ensemble(
            model_names=top_for_ensemble,
            ensemble_type="voting",
        )
        print(f"Ensemble Test ROC AUC: {ensemble_result.test_metrics.roc_auc:.4f}")
        print(f"Ensemble Test Accuracy: {ensemble_result.test_metrics.accuracy:.4f}")

    # Production readiness check
    print("\n" + "=" * 60)
    print("PRODUCTION READINESS CHECK")
    print("=" * 60)

    readiness = comparison.check_production_readiness(
        result.best_model,
        pipeline.X_test,
        pipeline.y_test,
        baseline_accuracy=baseline_acc,
    )

    print(f"\nBest model: {result.best_model_name}")
    print(f"Production ready: {readiness['production_ready']}")
    print("\nChecks:")
    for check, passed in readiness["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")

    print("\nRecommendations:")
    for rec in readiness["recommendations"]:
        print(f"  - {rec}")

    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        comparison.export_results(result, output_path)

        # Save best model
        pipeline.save_model(
            result.best_model,
            output_path / f"best_model_{result.best_model_name}.joblib",
        )

        print(f"\nResults saved to {output_path}")


def run_feature_analysis(
    data_filepath: str,
) -> None:
    """
    Run feature analysis to understand feature importance and relationships.
    """
    print("=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    # Load and prepare data
    pipeline = TrainingPipeline()
    df = pipeline.load_data(data_filepath)
    X, y = pipeline.prepare_data(df, add_derived_features=True)
    pipeline.split_data(X, y)

    engineer = FeatureEngineer()

    # Statistical feature selection
    print("\nStatistical Feature Ranking (F-test):")
    _, selected = engineer.select_features_statistical(
        pipeline.X_train, pipeline.y_train, k=10, method="f_classif"
    )
    print(engineer.feature_importances.head(15).to_string())

    # Mutual information
    print("\n\nMutual Information Ranking:")
    _, selected_mi = engineer.select_features_statistical(
        pipeline.X_train, pipeline.y_train, k=10, method="mutual_info"
    )
    print(engineer.feature_importances.head(15).to_string())

    # RFE with Random Forest
    print("\n\nRecursive Feature Elimination (Random Forest):")
    _, selected_rfe = engineer.select_features_rfe(
        pipeline.X_train, pipeline.y_train, n_features=10
    )
    print(f"Selected features: {selected_rfe}")

    # Correlation analysis
    print("\n\nHighly Correlated Features:")
    _, dropped = engineer.remove_correlated_features(
        pipeline.X_train, threshold=0.9
    )
    if dropped:
        print(f"Features to consider removing: {dropped}")
    else:
        print("No highly correlated features found (threshold=0.9)")


def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="NBA Game Prediction ML Experiments"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data CSV file",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for experiments",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory to save results",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment data with synthetic samples",
    )
    parser.add_argument(
        "--analyze-features",
        action="store_true",
        help="Run feature analysis",
    )

    args = parser.parse_args()

    if args.analyze_features and args.data:
        run_feature_analysis(args.data)
    elif args.synthetic:
        run_synthetic_experiment(
            n_samples=args.samples,
            output_dir=args.output,
            tune_hyperparameters=args.tune,
        )
    elif args.data:
        run_file_experiment(
            data_filepath=args.data,
            output_dir=args.output,
            tune_hyperparameters=args.tune,
            augment_data=args.augment,
        )
    else:
        print("Please specify --data <filepath> or --synthetic")
        print("Run with --help for more options")


if __name__ == "__main__":
    main()
