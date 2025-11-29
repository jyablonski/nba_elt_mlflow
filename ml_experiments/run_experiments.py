"""
Main experiment runner for NBA game predictions (V2).

Orchestrates the workflow:
1. Data Ingestion (Synthetic or File)
2. Temporal Splitting (Preventing Leakage)
3. Model Benchmarking (TimeSeries CV)
4. Financial Simulation (ROI Calculation)
5. Artifact Persistence

Usage:
    python -m ml_experiments.run_experiments --data data/nba_games_v2.csv --financial-check
    python -m ml_experiments.run_experiments --synthetic --samples 2000
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Optional

from ml_experiments.data_generator import SyntheticDataGenerator
from ml_experiments.models import ModelFactory, create_recommended_models
from ml_experiments.training_pipeline import TrainingPipeline
from ml_experiments.model_comparison import ModelComparison
from ml_experiments.feature_engineering import FeatureEngineer


def run_experiment_workflow(
    data_filepath: str,
    output_dir: Optional[str] = None,
    tune_hyperparameters: bool = False,
    run_financial_check: bool = False,
    date_col: str = "game_date",
    model_filter: Optional[str] = None
) -> None:
    """
    The Core Workflow: Agnostic to data source (Real vs Synthetic).
    """
    # 1. Initialize Components
    pipeline = TrainingPipeline(
        test_size=0.20,  # Test on last 20% of games
        cv_splits=5,  # 5 Time-series folds
        date_column=date_col,
    )
    comparison = ModelComparison()
    factory = ModelFactory()

    # 2. Load & Prepare Data (Strict Temporal Split)
    # This sets pipeline.X_train, pipeline.X_test, etc.
    pipeline.load_and_prep_data(data_filepath)

    print(f"\n{'=' * 60}")
    print("EXPERIMENT SETUP")
    print(f"{'=' * 60}")
    print(f"Training Samples: {len(pipeline.X_train)}")
    print(f"Testing Samples:  {len(pipeline.X_test)}")
    print(f"Features Used:    {len(pipeline.feature_names)}")

    # 3. Define Models to Evaluate
    all_models = create_recommended_models(factory)

    # --- FILTER LOGIC START ---
    if model_filter:
        if model_filter in all_models:
            print(f"Filtering to run only: {model_filter}")
            models = {model_filter: all_models[model_filter]}
        else:
            raise ValueError(f"Model '{model_filter}' not found. Available: {list(all_models.keys())}")
    else:
        models = all_models
    # --- FILTER LOGIC END ---
    print(f"Models selected: {list(models.keys())}")

    # 4. Run Comparison (TimeSeries Cross-Validation)
    print(f"\n{'=' * 60}")
    print("BENCHMARKING MODELS (TimeSeries CV)")
    print(f"{'=' * 60}")

    result = comparison.compare_models(
        models=models,
        X_train=pipeline.X_train,
        y_train=pipeline.y_train,
        X_test=pipeline.X_test,
        y_test=pipeline.y_test,
        cv_splits=5,
        use_time_series_split=True,
    )

    # Print Leaderboard
    print("\n>>> LEADERBOARD (Sorted by CV AUC) <<<")
    print(
        result.rankings[
            ["model", "cv_auc_mean", "test_auc", "test_acc", "test_brier"]
        ].to_string(index=False)
    )

    # 5. Hyperparameter Tuning (Optional)
    # If the user requested tuning, we take the winner and tune it.
    best_model_name = result.best_model_name
    best_model_obj = result.best_model

    if tune_hyperparameters:
        print(f"\n{'=' * 60}")
        print(f"TUNING BEST MODEL: {best_model_name}")
        print(f"{'=' * 60}")

        tune_result = pipeline.train_model(
            best_model_name,
            hyperparameter_tuning=True,
            tuning_method="random",
            n_iter=20,
        )
        best_model_obj = tune_result.best_model
        print(f"New Test AUC after tuning: {tune_result.test_metrics.auc:.4f}")
        print(f"Best Params: {tune_result.best_params}")

    # 6. Financial / Production Readiness Check
    print(f"\n{'=' * 60}")
    print("PRODUCTION READINESS & FINANCIAL SIMULATION")
    print(f"{'=' * 60}")

    # recover odds for the test set to calculate ROI
    test_odds = None
    if run_financial_check:
        # We need to reload the raw data to get the odds corresponding to the test indices
        # Since pipeline splits by simple index on sorted data:
        raw_df = pd.read_csv(data_filepath)
        if date_col in raw_df.columns:
            raw_df = raw_df.sort_values(date_col).reset_index(drop=True)

        split_idx = int(len(raw_df) * (1 - pipeline.test_size))
        test_odds = raw_df.iloc[split_idx:].reset_index(drop=True)

        # Verify alignment
        if len(test_odds) != len(pipeline.y_test):
            print("Warning: Odds dataframe length mismatch. Skipping financial check.")
            test_odds = None

    readiness = comparison.check_production_readiness(
        best_model_obj, pipeline.X_test, pipeline.y_test, test_odds_df=test_odds
    )

    print(f"\nSelected Model: {best_model_name}")
    print(f"Status: {'✅ READY' if readiness['production_ready'] else '❌ NOT READY'}")

    print("\nChecks:")
    for k, v in readiness["checks"].items():
        print(f" - {k:<25}: {v}")

    if "metrics" in readiness and "roi_simulation" in readiness["metrics"]:
        print("\n💰 Financial Simulation (Kelly Criterion):")
        print(f"   ROI: {readiness['metrics']['roi_simulation']:.2f}%")

    # 7. Save Artifacts
    if output_dir:
        out_path = Path(output_dir)
        comparison.export_results(result, out_path)

        # Create a result object to match pipeline expectation
        from ml_experiments.training_pipeline import TrainingResult

        final_result = TrainingResult(
            model_name=best_model_name,
            best_model=best_model_obj,
            best_params={},  # Populated if tuned
            cv_metrics={},
            test_metrics=readiness["metrics"],  # Use the final metrics
            feature_names=pipeline.feature_names,
            training_samples=len(pipeline.X_train),
        )

        pipeline.save_artifacts(final_result, out_path / "production_model.joblib")


def run_synthetic_experiment(samples: int, output_dir: str, tune: bool, model: str = None):
    """Generates data then runs the standard workflow."""
    print(f"Generating {samples} synthetic samples...")
    gen = SyntheticDataGenerator()
    df = gen.generate_samples(n_samples=samples)

    # Save to temp file to reuse the file-based workflow logic
    temp_path = "temp_synthetic_data.csv"
    df.to_csv(temp_path, index=False)

    try:
        run_experiment_workflow(
            data_filepath=temp_path,
            output_dir=output_dir,
            tune_hyperparameters=False,
            run_financial_check=True,
            model_filter=model  # Synthetic gen creates odds
        )
    finally:
        # Cleanup
        import os

        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_feature_analysis(data_filepath: str):
    """Standalone analysis of feature importance."""
    print("Running Feature Analysis...")
    pipeline = TrainingPipeline()
    pipeline.load_and_prep_data(data_filepath)

    engineer = FeatureEngineer()

    print("\n1. Mutual Information (Non-linear relationships):")
    _, top_mi = engineer.select_features_statistical(
        pipeline.X_train, pipeline.y_train, method="mutual_info", k=10
    )
    print(engineer.feature_importances.head(10))

    print("\n2. Recursive Feature Elimination (Random Forest):")
    _, top_rfe = engineer.select_features_rfe(
        pipeline.X_train, pipeline.y_train, n_features=10
    )
    print(f"Top 10 RFE: {top_rfe}")


def main():
    parser = argparse.ArgumentParser(description="NBA Prediction Experiment Runner")

    # Mode Selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", type=str, help="Path to real data CSV")
    group.add_argument(
        "--synthetic", action="store_true", help="Run with synthetic data"
    )
    group.add_argument(
        "--analyze-features", type=str, help="Run feature analysis on this file"
    )

    # Options
    parser.add_argument(
        "--samples", type=int, default=2000, help="Num synthetic samples"
    )
    parser.add_argument(
        "--out", type=str, default="./experiment_results", help="Output directory"
    )
    parser.add_argument("--tune", action="store_true", help="Enable hyperparam tuning")
    parser.add_argument(
        "--financial",
        action="store_true",
        help="Run betting simulation (requires moneyline cols)",
    )
    parser.add_argument("--model", type=str, help="Specific model to run (e.g. logistic_regression)")

    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_experiment(args.samples, args.out, args.tune, args.model)
    elif args.analyze_features:
        run_feature_analysis(args.analyze_features)
    elif args.data:
        run_experiment_workflow(
            data_filepath=args.data,
            output_dir=args.out,
            tune_hyperparameters=args.tune,
            run_financial_check=args.financial,
            model_filter=args.model
        )


if __name__ == "__main__":
    main()
