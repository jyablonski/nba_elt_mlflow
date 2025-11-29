"""
Training pipeline for NBA game prediction models.

Provides end-to-end training capabilities enforcing Time-Series logic
to prevent data leakage (Look-ahead bias).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, List, Union

import pandas as pd
import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from ml_experiments.config import (
    TARGET_COLUMN,
    METADATA_COLUMNS,
)
from ml_experiments.feature_engineering import FeatureEngineer
from ml_experiments.models import ModelFactory
from ml_experiments.evaluation import ModelEvaluator, EvaluationMetrics


@dataclass
class TrainingResult:
    """Results from a training run."""

    model_name: str
    best_model: BaseEstimator
    best_params: Dict[str, Any]
    cv_metrics: Dict[str, float]
    test_metrics: EvaluationMetrics
    feature_names: List[str]
    training_samples: int


class TrainingPipeline:
    """
    End-to-end training pipeline for NBA game predictions.
    """

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.20,  # 20% of most recent games for testing
        cv_splits: int = 5,
        date_column: str = "game_date",
    ):
        self.random_state = random_state
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.date_column = date_column

        self.factory = ModelFactory()
        self.engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()

        # State storage
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_names: List[str] = []

    def load_and_prep_data(
        self,
        filepath: Union[str, Path],
        target_column: str = TARGET_COLUMN,
    ) -> None:
        """
        Load data, sort by date, split, and preprocess.

        Best Practice:
        We split raw data FIRST, then apply feature engineering separately
        to Train and Test to ensure no data leakage (imputation values, scaling).
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)

        # 1. Sort by Date (Crucial for Time Series)
        if self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            df = df.sort_values(self.date_column).reset_index(drop=True)
        else:
            print(
                f"Warning: {self.date_column} not found. Assuming data is already sorted."
            )

        # 2. Encode Target
        if target_column in df.columns:
            # Handle W/L text or boolean
            if df[target_column].dtype == object:
                df[target_column] = df[target_column].apply(
                    lambda x: 1 if str(x).upper() == "W" else 0
                )
        else:
            raise ValueError(f"Target column '{target_column}' not found")

        split_idx = int(len(df) * (1 - self.test_size))

        train_raw = df.iloc[:split_idx].copy()
        test_raw = df.iloc[split_idx:].copy()

        print(
            f"Temporal Split: Train on {len(train_raw)} oldest games, Test on {len(test_raw)} newest games."
        )

        # 4. Feature Engineering (Fit on Train, Transform Test)
        # Note: We keep metadata in raw dfs for later analysis, but strip them for X matrices
        self.X_train = self.engineer.preprocess_data(train_raw, is_training=True)
        self.X_test = self.engineer.preprocess_data(test_raw, is_training=False)

        # 5. Separate Targets
        self.y_train = train_raw[target_column]
        self.y_test = test_raw[target_column]

        # 6. Drop Metadata & Target from Features
        drop_cols = METADATA_COLUMNS + [target_column]

        self.X_train = self.X_train.drop(
            columns=[c for c in drop_cols if c in self.X_train.columns]
        )
        self.X_test = self.X_test.drop(
            columns=[c for c in drop_cols if c in self.X_test.columns]
        )

        self.feature_names = self.X_train.columns.tolist()
        print(f"Final Feature Count: {len(self.feature_names)}")

    def train_model(
        self,
        model_name: str,
        hyperparameter_tuning: bool = False,
        tuning_method: str = "random",
        n_iter: int = 20,
        calibrate: bool = False,
    ) -> TrainingResult:
        """
        Train a single model with TimeSeries CV and optional Tuning/Calibration.
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_prep_data first.")

        config = self.factory.get_model_config(model_name)

        # Best Practice: Use TimeSeriesSplit for CV
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        best_model = None
        best_params = config.default_params

        # 1. Hyperparameter Tuning
        if hyperparameter_tuning and config.hyperparameter_space:
            print(f"Tuning {model_name}...")
            base_estimator = self.factory.create_model(model_name, wrap_pipeline=True)

            # Helper to prefix params for Pipeline (e.g., 'model__C' instead of 'C')
            # The Factory wraps models in a 'model' step if they need scaling.
            search_space = config.hyperparameter_space
            if isinstance(base_estimator, Pipeline):
                search_space = {f"model__{k}": v for k, v in search_space.items()}

            if tuning_method == "grid":
                search = GridSearchCV(
                    base_estimator, search_space, cv=tscv, scoring="roc_auc", n_jobs=-1
                )
            else:
                search = RandomizedSearchCV(
                    base_estimator,
                    search_space,
                    n_iter=n_iter,
                    cv=tscv,
                    scoring="roc_auc",
                    n_jobs=-1,
                    random_state=self.random_state,
                )

            search.fit(self.X_train, self.y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
        else:
            best_model = self.factory.create_model(model_name, wrap_pipeline=True)
            best_model.fit(self.X_train, self.y_train)

        # 2. Calibration (Optional but recommended for Betting)
        if calibrate and config.supports_proba:
            print(f"Calibrating {model_name}...")
            # We must use a fresh splitter or prefit.
            # Using CalibratedClassifierCV with cv=tscv fits on folds and averages.
            best_model = CalibratedClassifierCV(best_model, cv=tscv, method="isotonic")
            best_model.fit(self.X_train, self.y_train)

        # 3. Cross-Validation Metrics (on Train set)
        cv_results = cross_validate(
            best_model,
            self.X_train,
            self.y_train,
            cv=tscv,
            scoring=["accuracy", "roc_auc", "neg_brier_score"],
        )

        cv_metrics = {
            "cv_accuracy": cv_results["test_accuracy"].mean(),
            "cv_auc": cv_results["test_roc_auc"].mean(),
            "cv_brier": -cv_results["test_neg_brier_score"].mean(),
        }

        # 4. Final Test Set Evaluation
        test_metrics = self.evaluator.evaluate(
            best_model, self.X_test, self.y_test, model_name=model_name
        )

        return TrainingResult(
            model_name=model_name,
            best_model=best_model,
            best_params=best_params,
            cv_metrics=cv_metrics,
            test_metrics=test_metrics,
            feature_names=self.feature_names,
            training_samples=len(self.X_train),
        )

    def run_experiment(
        self,
        data_filepath: Union[str, Path],
        models_to_run: List[str] = ["logistic_regression", "xgboost"],
        tune_hyperparams: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, TrainingResult]:
        """
        Run a full experiment: Load -> Split -> Train Multiple -> Save Best.
        """
        self.load_and_prep_data(data_filepath)

        results = {}

        for name in models_to_run:
            if not self.factory.is_model_available(name):
                print(f"Skipping {name} (dependencies not installed).")
                continue

            print(f"\n--- Training {name} ---")
            try:
                res = self.train_model(name, hyperparameter_tuning=tune_hyperparams)
                results[name] = res
                print(f"Test AUC: {res.test_metrics.auc:.4f}")
            except Exception as e:
                print(f"Failed to train {name}: {e}")

        # Determine Best Model (by AUC)
        if results:
            best_name = max(results, key=lambda x: results[x].test_metrics.auc)
            print(
                f"\n🏆 Best Model: {best_name} (AUC: {results[best_name].test_metrics.auc:.4f})"
            )

            if output_dir:
                self.save_artifacts(
                    results[best_name], Path(output_dir) / "best_model.joblib"
                )

        return results

    def save_artifacts(self, result: TrainingResult, filepath: Path) -> None:
        """
        Save the model AND the feature engineering state.
        Required for consistent inference later.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # FIX: Handle case where test_metrics is already a dict (from financial check)
        # vs an EvaluationMetrics object (from standard training)
        metrics_data = (
            result.test_metrics.to_dict()
            if hasattr(result.test_metrics, "to_dict")
            else result.test_metrics
        )

        payload = {
            "model": result.best_model,
            "feature_engineer": self.engineer,  # Contains Imputer stats
            "feature_names": self.feature_names,
            "metrics": metrics_data,
            "params": result.best_params,
        }

        joblib.dump(payload, filepath)
        print(f"Pipeline artifacts saved to {filepath}")

    @staticmethod
    def load_artifacts(filepath: Path) -> Dict[str, Any]:
        """
        Load a saved pipeline payload without instantiating the class.
        """
        # Ensure pathlib Path object
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Artifact not found at {filepath}")

        return joblib.load(filepath)
