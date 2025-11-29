"""
Model factory and definitions for NBA game predictions.

Provides a unified interface for creating, configuring, and persisting
machine learning models, enforcing best practices like Pipelining.
"""

import importlib
import joblib
import os
from typing import Any, Optional, Union

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from ml_experiments.config import MODEL_CONFIGS, ModelConfig


class ModelFactory:
    """
    Factory for creating machine learning models.

    Handles:
    - Dynamic instantiation of models (sklearn, xgboost, etc.)
    - Automatic Pipeline creation (scaling + modeling)
    - Ensemble generation
    - Model persistence (save/load)
    """

    def __init__(self):
        """Initialize the model factory."""
        self._model_cache: dict[str, type] = {}

    def get_available_models(self) -> list[str]:
        """Get list of available model names defined in config."""
        return list(MODEL_CONFIGS.keys())

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {self.get_available_models()}"
            )
        return MODEL_CONFIGS[model_name]

    def create_model(
        self,
        model_name: str,
        params: Optional[dict[str, Any]] = None,
        wrap_pipeline: bool = True,
    ) -> Union[BaseEstimator, Pipeline]:
        """
        Create a model instance, optionally wrapped in a preprocessing pipeline.

        Best Practice:
        If the model requires scaling (e.g., LogisticRegression, SVM),
        this returns a Pipeline([('scaler', StandardScaler), ('model', Obj)]).
        This prevents data leakage during cross-validation.

        Args:
            model_name: Name of the model to create
            params: Optional parameters to override defaults
            wrap_pipeline: Whether to wrap in a Pipeline if config requires it

        Returns:
            Instantiated model or Pipeline
        """
        config = self.get_model_config(model_name)
        model_class = self._get_model_class(config.model_class)

        # 1. Merge default params with provided params
        final_params = {**config.default_params}
        if params:
            final_params.update(params)

        # 2. Instantiate the base estimator
        # We instantiate here so we don't have to deal with 'model__param' naming
        # prefixes if we were to pass params to the Pipeline constructor directly.
        estimator = model_class(**final_params)

        # 3. Wrap in Pipeline if required (Best Practice)
        if wrap_pipeline and config.requires_scaling:
            steps = [
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
            return Pipeline(steps)

        return estimator

    def _get_model_class(self, class_path: str) -> type:
        """Import and return a model class from its full path."""
        if class_path in self._model_cache:
            return self._model_cache[class_path]

        module_path, class_name = class_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            self._model_cache[class_path] = model_class
            return model_class
        except ImportError as e:
            raise ImportError(
                f"Could not import {class_path}. "
                f"Make sure the required package (e.g., xgboost, lightgbm) is installed.\nError: {e}"
            )

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model can be instantiated (dependencies available)."""
        if model_name not in MODEL_CONFIGS:
            return False
        config = MODEL_CONFIGS[model_name]
        try:
            self._get_model_class(config.model_class)
            return True
        except ImportError:
            return False

    def get_installed_models(self) -> list[str]:
        """Get list of models whose dependencies are installed."""
        return [name for name in MODEL_CONFIGS if self.is_model_available(name)]

    # --- Ensemble Methods ---

    def create_voting_ensemble(
        self,
        model_names: list[str],
        voting: str = "soft",
        weights: Optional[list[float]] = None,
    ) -> VotingClassifier:
        """Create a VotingClassifier from multiple named models."""
        estimators = [(name, self.create_model(name)) for name in model_names]
        return VotingClassifier(estimators=estimators, voting=voting, weights=weights)

    def create_stacking_ensemble(
        self,
        model_names: list[str],
        final_estimator: Optional[BaseEstimator] = None,
        cv: int = 5,
    ) -> StackingClassifier:
        """Create a StackingClassifier (Level 1 learner)."""
        estimators = [(name, self.create_model(name)) for name in model_names]

        if final_estimator is None:
            final_estimator = LogisticRegression(random_state=42)

        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            passthrough=False,
        )

    def create_bagging_ensemble(
        self,
        base_model_name: str,
        n_estimators: int = 10,
        random_state: int = 42,
        **kwargs,
    ) -> BaggingClassifier:
        """
        Create a BaggingClassifier (Bootstrap Aggregating).

        Uses sklearn's native implementation which is optimized for performance
        and supports parallel processing.
        """
        base_estimator = self.create_model(base_model_name)

        return BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,  # Parallelize by default
            **kwargs,
        )

    # --- Persistence Methods ---

    def save_model(self, model: BaseEstimator, filepath: str) -> None:
        """Save a trained model to disk."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(model, filepath)

    def load_model(self, filepath: str) -> BaseEstimator:
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)


class CalibratedModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to easily calibrate any model from the factory.

    This is preferred over manual averaging as it allows the use
    of sklearn's CalibratedClassifierCV on top of Pipelines.
    """

    def __init__(
        self, base_estimator: BaseEstimator, method: str = "isotonic", cv: int = 5
    ):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.calibrated_classifier_ = None

    def fit(self, X, y):
        self.calibrated_classifier_ = CalibratedClassifierCV(
            self.base_estimator, method=self.method, cv=self.cv
        )
        self.calibrated_classifier_.fit(X, y)
        self.classes_ = self.calibrated_classifier_.classes_
        return self

    def predict(self, X):
        return self.calibrated_classifier_.predict(X)

    def predict_proba(self, X):
        return self.calibrated_classifier_.predict_proba(X)


def create_recommended_models(factory: ModelFactory) -> dict[str, BaseEstimator]:
    """
    Create a set of recommended models for NBA game prediction.

    Returns:
        Dictionary mapping model names to model instances
    """
    installed = factory.get_installed_models()
    models = {}

    # Core models
    if "logistic_regression" in installed:
        models["logistic_regression"] = factory.create_model("logistic_regression")

    if "random_forest" in installed:
        models["random_forest"] = factory.create_model("random_forest")

    if "gradient_boosting" in installed:
        models["gradient_boosting"] = factory.create_model("gradient_boosting")

    # Advanced Tree models (High performance)
    if "xgboost" in installed:
        models["xgboost"] = factory.create_model("xgboost")

    if "lightgbm" in installed:
        models["lightgbm"] = factory.create_model("lightgbm")

    # Neural Net (Good for capturing non-linear feature interactions)
    if "mlp" in installed:
        models["mlp"] = factory.create_model("mlp")

    return models
