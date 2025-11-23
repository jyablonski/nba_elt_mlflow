"""
Training pipeline for NBA game prediction models.

Provides end-to-end training capabilities including data preprocessing,
cross-validation, hyperparameter tuning, and model persistence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_experiments.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    METADATA_COLUMNS,
    MODEL_CONFIGS,
    CV_CONFIG,
    TRAINING_CONFIG,
)
from ml_experiments.feature_engineering import FeatureEngineer
from ml_experiments.models import ModelFactory
from ml_experiments.evaluation import ModelEvaluator, EvaluationMetrics


@dataclass
class TrainingResult:
    """Results from a training run."""

    model_name: str
    best_model: BaseEstimator
    best_params: dict[str, Any]
    cv_scores: np.ndarray
    test_metrics: EvaluationMetrics
    feature_names: list[str]
    training_samples: int


class TrainingPipeline:
    """
    End-to-end training pipeline for NBA game predictions.

    Handles data loading, preprocessing, feature engineering,
    model training, hyperparameter tuning, and evaluation.
    """

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ):
        """
        Initialize the training pipeline.

        Args:
            random_state: Random seed for reproducibility
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
        """
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        self.factory = ModelFactory()
        self.engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_names: list[str] = []

    def load_data(
        self,
        filepath: str | Path,
        target_column: str = TARGET_COLUMN,
        exclude_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load and prepare data from a CSV file.

        Args:
            filepath: Path to CSV file
            target_column: Name of target column
            exclude_columns: Columns to exclude (defaults to METADATA_COLUMNS)

        Returns:
            Prepared DataFrame
        """
        df = pd.read_csv(filepath)

        # Convert outcome if necessary
        if target_column in df.columns and df[target_column].dtype == object:
            df[target_column] = df[target_column].replace({"W": 1, "L": 0})

        # Drop metadata columns that exist in the dataframe
        if exclude_columns is None:
            exclude_columns = METADATA_COLUMNS

        cols_to_drop = [col for col in exclude_columns if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = TARGET_COLUMN,
        add_derived_features: bool = True,
        scale_features: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            add_derived_features: Whether to add engineered features
            scale_features: Whether to scale features

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        # Separate features and target
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])

        # Add derived features
        if add_derived_features:
            X = self.engineer.create_derived_features(X)

        # Scale if requested
        if scale_features:
            X = self.engineer.scale_features(X, method="standard", fit=True)

        self.feature_names = list(X.columns)
        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratify: bool = True,
    ) -> None:
        """
        Split data into train and test sets.

        Args:
            X: Feature DataFrame
            y: Target Series
            stratify: Whether to use stratified splitting
        """
        stratify_arg = y if stratify else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_arg,
        )

    def train_model(
        self,
        model_name: str,
        hyperparameter_tuning: bool = False,
        tuning_method: str = "random",
        n_iter: int = 50,
    ) -> TrainingResult:
        """
        Train a single model with optional hyperparameter tuning.

        Args:
            model_name: Name of model to train
            hyperparameter_tuning: Whether to perform HP tuning
            tuning_method: 'grid' or 'random' search
            n_iter: Number of iterations for random search

        Returns:
            TrainingResult with trained model and metrics
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call split_data first.")

        config = self.factory.get_model_config(model_name)

        if hyperparameter_tuning and config.hyperparameter_space:
            best_model, best_params = self._tune_hyperparameters(
                model_name, tuning_method, n_iter
            )
        else:
            best_model = self.factory.create_model(model_name)
            best_model.fit(self.X_train, self.y_train)
            best_params = config.default_params

        # Cross-validation scores
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        cv_scores = cross_val_score(
            best_model, self.X_train, self.y_train, cv=cv, scoring="accuracy"
        )

        # Test set evaluation
        test_metrics = self.evaluator.evaluate(
            best_model,
            self.X_test,
            self.y_test,
            model_name=model_name,
            cv_scores=cv_scores,
        )

        return TrainingResult(
            model_name=model_name,
            best_model=best_model,
            best_params=best_params,
            cv_scores=cv_scores,
            test_metrics=test_metrics,
            feature_names=self.feature_names,
            training_samples=len(self.X_train),
        )

    def _tune_hyperparameters(
        self,
        model_name: str,
        method: str = "random",
        n_iter: int = 50,
    ) -> tuple[BaseEstimator, dict[str, Any]]:
        """Perform hyperparameter tuning."""
        config = self.factory.get_model_config(model_name)
        base_model = self.factory.create_model(model_name)

        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        if method == "grid":
            search = GridSearchCV(
                base_model,
                config.hyperparameter_space,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                config.hyperparameter_space,
                n_iter=n_iter,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )

        search.fit(self.X_train, self.y_train)

        return search.best_estimator_, search.best_params_

    def train_all_models(
        self,
        hyperparameter_tuning: bool = False,
        model_names: Optional[list[str]] = None,
    ) -> dict[str, TrainingResult]:
        """
        Train all available models.

        Args:
            hyperparameter_tuning: Whether to tune hyperparameters
            model_names: Optional list of specific models to train

        Returns:
            Dictionary of model_name -> TrainingResult
        """
        if model_names is None:
            model_names = self.factory.get_installed_models()

        results = {}

        for name in model_names:
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print('='*50)

            try:
                result = self.train_model(
                    name,
                    hyperparameter_tuning=hyperparameter_tuning,
                )
                results[name] = result

                print(f"\nResults for {name}:")
                print(result.test_metrics.summary())

            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        return results

    def create_ensemble(
        self,
        model_names: list[str],
        ensemble_type: str = "voting",
        weights: Optional[list[float]] = None,
    ) -> TrainingResult:
        """
        Create and train an ensemble model.

        Args:
            model_names: Models to include in ensemble
            ensemble_type: 'voting' or 'stacking'
            weights: Optional weights for voting ensemble

        Returns:
            TrainingResult with trained ensemble
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call split_data first.")

        if ensemble_type == "voting":
            ensemble = self.factory.create_voting_ensemble(
                model_names, voting="soft", weights=weights
            )
        elif ensemble_type == "stacking":
            ensemble = self.factory.create_stacking_ensemble(
                model_names, cv=self.cv_folds
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

        ensemble.fit(self.X_train, self.y_train)

        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        cv_scores = cross_val_score(
            ensemble, self.X_train, self.y_train, cv=cv, scoring="accuracy"
        )

        # Test evaluation
        test_metrics = self.evaluator.evaluate(
            ensemble,
            self.X_test,
            self.y_test,
            model_name=f"{ensemble_type}_ensemble",
            cv_scores=cv_scores,
        )

        return TrainingResult(
            model_name=f"{ensemble_type}_ensemble",
            best_model=ensemble,
            best_params={"models": model_names, "type": ensemble_type},
            cv_scores=cv_scores,
            test_metrics=test_metrics,
            feature_names=self.feature_names,
            training_samples=len(self.X_train),
        )

    def save_model(
        self,
        model: BaseEstimator,
        filepath: str | Path,
        include_metadata: bool = True,
    ) -> None:
        """
        Save a trained model to disk.

        Args:
            model: Trained model to save
            filepath: Output path
            include_metadata: Whether to save feature names etc.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if include_metadata:
            package = {
                "model": model,
                "feature_names": self.feature_names,
                "scaler": self.engineer.scaler,
            }
            dump(package, filepath)
        else:
            dump(model, filepath)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str | Path) -> BaseEstimator:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model
        """
        loaded = load(filepath)

        if isinstance(loaded, dict):
            self.feature_names = loaded.get("feature_names", [])
            self.engineer.scaler = loaded.get("scaler")
            return loaded["model"]
        else:
            return loaded

    def get_feature_importance(
        self,
        model: BaseEstimator,
    ) -> pd.DataFrame:
        """
        Get feature importance from a trained model.

        Args:
            model: Trained model

        Returns:
            DataFrame with feature importances
        """
        return self.engineer.get_feature_importance(model, self.feature_names)

    def run_full_pipeline(
        self,
        data_filepath: str | Path,
        output_filepath: Optional[str | Path] = None,
        hyperparameter_tuning: bool = True,
        model_names: Optional[list[str]] = None,
    ) -> dict[str, TrainingResult]:
        """
        Run the complete training pipeline.

        Args:
            data_filepath: Path to training data CSV
            output_filepath: Optional path to save best model
            hyperparameter_tuning: Whether to tune hyperparameters
            model_names: Optional list of models to train

        Returns:
            Dictionary of training results
        """
        print("Loading data...")
        df = self.load_data(data_filepath)

        print("Preparing features...")
        X, y = self.prepare_data(df, add_derived_features=True)

        print("Splitting data...")
        self.split_data(X, y)
        print(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

        print("\nTraining models...")
        results = self.train_all_models(
            hyperparameter_tuning=hyperparameter_tuning,
            model_names=model_names,
        )

        # Find best model
        if results:
            best_name = max(
                results.keys(),
                key=lambda k: results[k].test_metrics.roc_auc
            )
            best_result = results[best_name]

            print(f"\n{'='*50}")
            print(f"Best Model: {best_name}")
            print(f"ROC AUC: {best_result.test_metrics.roc_auc:.4f}")
            print('='*50)

            if output_filepath:
                self.save_model(best_result.best_model, output_filepath)

        return results


def create_sklearn_pipeline(
    model: BaseEstimator,
    scale: bool = True,
) -> Pipeline:
    """
    Create a sklearn Pipeline with optional scaling.

    Args:
        model: Model to include in pipeline
        scale: Whether to include StandardScaler

    Returns:
        sklearn Pipeline
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))

    return Pipeline(steps)
