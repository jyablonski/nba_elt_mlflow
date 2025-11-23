"""
Model factory and definitions for NBA game predictions.

Provides a unified interface for creating and configuring
various machine learning models.
"""

import importlib
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from ml_experiments.config import MODEL_CONFIGS, ModelConfig


class ModelFactory:
    """
    Factory for creating machine learning models.

    Provides a unified interface for instantiating models from
    various libraries (sklearn, xgboost, lightgbm, etc.).
    """

    def __init__(self):
        """Initialize the model factory."""
        self._model_cache: dict[str, type] = {}

    def get_available_models(self) -> list[str]:
        """Get list of available model names."""
        return list(MODEL_CONFIGS.keys())

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {self.get_available_models()}"
            )
        return MODEL_CONFIGS[model_name]

    def create_model(
        self,
        model_name: str,
        params: Optional[dict[str, Any]] = None,
    ) -> BaseEstimator:
        """
        Create a model instance with specified parameters.

        Args:
            model_name: Name of the model to create
            params: Optional parameters to override defaults

        Returns:
            Instantiated model
        """
        config = self.get_model_config(model_name)

        # Get model class
        model_class = self._get_model_class(config.model_class)

        # Merge default params with provided params
        model_params = {**config.default_params}
        if params:
            model_params.update(params)

        return model_class(**model_params)

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
                f"Make sure the required package is installed. Error: {e}"
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

    def create_voting_ensemble(
        self,
        model_names: list[str],
        voting: str = "soft",
        weights: Optional[list[float]] = None,
    ) -> VotingClassifier:
        """
        Create a voting ensemble from multiple models.

        Args:
            model_names: List of model names to include
            voting: Voting type ('hard' or 'soft')
            weights: Optional weights for each model

        Returns:
            VotingClassifier instance
        """
        estimators = [(name, self.create_model(name)) for name in model_names]

        return VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
        )

    def create_stacking_ensemble(
        self,
        model_names: list[str],
        final_estimator: Optional[BaseEstimator] = None,
        cv: int = 5,
    ) -> StackingClassifier:
        """
        Create a stacking ensemble from multiple models.

        Args:
            model_names: List of model names for base estimators
            final_estimator: Meta-learner (default: LogisticRegression)
            cv: Number of cross-validation folds

        Returns:
            StackingClassifier instance
        """
        estimators = [(name, self.create_model(name)) for name in model_names]

        if final_estimator is None:
            final_estimator = LogisticRegression(random_state=42, max_iter=1000)

        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            passthrough=False,
        )


class CalibratedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble that calibrates probability predictions.

    Combines multiple models and calibrates their probabilities
    using isotonic regression or Platt scaling.
    """

    def __init__(
        self,
        base_models: list[tuple[str, BaseEstimator]],
        calibration_method: str = "isotonic",
        cv: int = 5,
    ):
        """
        Initialize the calibrated ensemble.

        Args:
            base_models: List of (name, model) tuples
            calibration_method: 'isotonic' or 'sigmoid'
            cv: Number of CV folds for calibration
        """
        self.base_models = base_models
        self.calibration_method = calibration_method
        self.cv = cv
        self.calibrated_models_ = []
        self.classes_ = None

    def fit(self, X, y):
        """Fit and calibrate all base models."""
        from sklearn.calibration import CalibratedClassifierCV

        self.classes_ = np.unique(y)
        self.calibrated_models_ = []

        for name, model in self.base_models:
            calibrated = CalibratedClassifierCV(
                model,
                method=self.calibration_method,
                cv=self.cv,
            )
            calibrated.fit(X, y)
            self.calibrated_models_.append((name, calibrated))

        return self

    def predict_proba(self, X):
        """Average probability predictions from all calibrated models."""
        probas = np.array(
            [model.predict_proba(X) for _, model in self.calibrated_models_]
        )
        return np.mean(probas, axis=0)

    def predict(self, X):
        """Predict class labels based on averaged probabilities."""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


class BaggingModelEnsemble(BaseEstimator, ClassifierMixin):
    """
    Custom bagging ensemble that trains multiple instances
    of the same model on bootstrap samples.
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        n_estimators: int = 10,
        sample_ratio: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initialize the bagging ensemble.

        Args:
            base_model: Model to use as base estimator
            n_estimators: Number of models to train
            sample_ratio: Ratio of samples for each bootstrap
            random_state: Random seed
        """
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.models_ = []
        self.classes_ = None

    def fit(self, X, y):
        """Train multiple models on bootstrap samples."""
        from sklearn.base import clone
        from sklearn.utils import resample

        self.classes_ = np.unique(y)
        self.models_ = []

        n_samples = int(len(X) * self.sample_ratio)
        rng = np.random.default_rng(self.random_state)

        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_boot, y_boot = resample(
                X, y, n_samples=n_samples, random_state=rng.integers(10000)
            )

            # Clone and fit model
            model = clone(self.base_model)
            model.fit(X_boot, y_boot)
            self.models_.append(model)

        return self

    def predict_proba(self, X):
        """Average probability predictions from all models."""
        probas = np.array([model.predict_proba(X) for model in self.models_])
        return np.mean(probas, axis=0)

    def predict(self, X):
        """Predict class labels based on averaged probabilities."""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def create_baseline_model() -> LogisticRegression:
    """Create the baseline logistic regression model for comparison."""
    return LogisticRegression(random_state=42, max_iter=1000)


def create_recommended_models(factory: ModelFactory) -> dict[str, BaseEstimator]:
    """
    Create a set of recommended models for NBA game prediction.

    These models have been selected for their performance characteristics
    and suitability for binary classification with probability outputs.

    Args:
        factory: ModelFactory instance

    Returns:
        Dictionary mapping model names to model instances
    """
    installed = factory.get_installed_models()

    models = {}

    # Always include baseline
    models["baseline_logistic"] = create_baseline_model()

    # Core sklearn models (always available)
    models["logistic_regression"] = factory.create_model("logistic_regression")
    models["random_forest"] = factory.create_model("random_forest")
    models["gradient_boosting"] = factory.create_model("gradient_boosting")

    # Extra models if sklearn components available
    if "extra_trees" in installed:
        models["extra_trees"] = factory.create_model("extra_trees")

    if "mlp" in installed:
        models["mlp"] = factory.create_model("mlp")

    # XGBoost and LightGBM if available
    if "xgboost" in installed:
        models["xgboost"] = factory.create_model("xgboost")

    if "lightgbm" in installed:
        models["lightgbm"] = factory.create_model("lightgbm")

    return models
