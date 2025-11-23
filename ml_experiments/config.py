"""
Configuration and constants for NBA ML experiments.

Defines feature columns, target variable, model configurations,
and hyperparameter search spaces.
"""

from dataclasses import dataclass, field
from typing import Any

# Feature column definitions
FEATURE_COLUMNS = [
    "home_team_rank",
    "home_days_rest",
    "home_team_avg_pts_scored",
    "home_team_avg_pts_scored_opp",
    "home_team_win_pct",
    "home_team_win_pct_last10",
    "home_is_top_players",
    "away_team_rank",
    "away_days_rest",
    "away_team_avg_pts_scored",
    "away_team_avg_pts_scored_opp",
    "away_team_win_pct",
    "away_team_win_pct_last10",
    "away_is_top_players",
]

# Columns that need to be excluded when loading data for training
METADATA_COLUMNS = [
    "home_team",
    "away_team",
    "home_moneyline",
    "away_moneyline",
    "game_date",
    "home_team_predicted_win_pct",
    "away_team_predicted_win_pct",
    "ml_accuracy",
    "ml_money_col",
    "home_implied_probability",
    "away_implied_probability",
    "ml_prediction",
    "actual_outcome",
    "proper_date",
]

TARGET_COLUMN = "outcome"

# Feature groups for easier management
HOME_FEATURES = [f for f in FEATURE_COLUMNS if f.startswith("home_")]
AWAY_FEATURES = [f for f in FEATURE_COLUMNS if f.startswith("away_")]

# Feature statistics for synthetic data generation (based on typical NBA data)
FEATURE_STATS = {
    "home_team_rank": {"min": 1, "max": 30, "mean": 15.5, "std": 8.7},
    "away_team_rank": {"min": 1, "max": 30, "mean": 15.5, "std": 8.7},
    "home_days_rest": {"min": 0, "max": 7, "mean": 1.5, "std": 1.2},
    "away_days_rest": {"min": 0, "max": 7, "mean": 1.5, "std": 1.2},
    "home_team_avg_pts_scored": {"min": 100, "max": 130, "mean": 112, "std": 5},
    "away_team_avg_pts_scored": {"min": 100, "max": 130, "mean": 112, "std": 5},
    "home_team_avg_pts_scored_opp": {"min": 100, "max": 130, "mean": 112, "std": 5},
    "away_team_avg_pts_scored_opp": {"min": 100, "max": 130, "mean": 112, "std": 5},
    "home_team_win_pct": {"min": 0.1, "max": 0.9, "mean": 0.5, "std": 0.15},
    "away_team_win_pct": {"min": 0.1, "max": 0.9, "mean": 0.5, "std": 0.15},
    "home_team_win_pct_last10": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.2},
    "away_team_win_pct_last10": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.2},
    "home_is_top_players": {"min": 0, "max": 2, "values": [0, 1, 2], "probs": [0.1, 0.2, 0.7]},
    "away_is_top_players": {"min": 0, "max": 2, "values": [0, 1, 2], "probs": [0.1, 0.2, 0.7]},
}


@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""

    name: str
    model_class: str
    default_params: dict[str, Any] = field(default_factory=dict)
    hyperparameter_space: dict[str, Any] = field(default_factory=dict)
    requires_scaling: bool = False
    supports_proba: bool = True


# Model configurations with hyperparameter search spaces
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "logistic_regression": ModelConfig(
        name="Logistic Regression",
        model_class="sklearn.linear_model.LogisticRegression",
        default_params={"random_state": 42, "max_iter": 1000},
        hyperparameter_space={
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        },
        requires_scaling=True,
        supports_proba=True,
    ),
    "random_forest": ModelConfig(
        name="Random Forest",
        model_class="sklearn.ensemble.RandomForestClassifier",
        default_params={"random_state": 42, "n_jobs": -1},
        hyperparameter_space={
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
        requires_scaling=False,
        supports_proba=True,
    ),
    "gradient_boosting": ModelConfig(
        name="Gradient Boosting",
        model_class="sklearn.ensemble.GradientBoostingClassifier",
        default_params={"random_state": 42},
        hyperparameter_space={
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0],
        },
        requires_scaling=False,
        supports_proba=True,
    ),
    "xgboost": ModelConfig(
        name="XGBoost",
        model_class="xgboost.XGBClassifier",
        default_params={
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "n_jobs": -1,
        },
        hyperparameter_space={
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "gamma": [0, 0.1, 0.2],
        },
        requires_scaling=False,
        supports_proba=True,
    ),
    "lightgbm": ModelConfig(
        name="LightGBM",
        model_class="lightgbm.LGBMClassifier",
        default_params={"random_state": 42, "n_jobs": -1, "verbose": -1},
        hyperparameter_space={
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [-1, 5, 10, 15],
            "num_leaves": [15, 31, 63, 127],
            "min_child_samples": [5, 10, 20],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
        requires_scaling=False,
        supports_proba=True,
    ),
    "svm": ModelConfig(
        name="Support Vector Machine",
        model_class="sklearn.svm.SVC",
        default_params={"random_state": 42, "probability": True},
        hyperparameter_space={
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "poly", "linear"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
        requires_scaling=True,
        supports_proba=True,
    ),
    "mlp": ModelConfig(
        name="Neural Network (MLP)",
        model_class="sklearn.neural_network.MLPClassifier",
        default_params={"random_state": 42, "max_iter": 1000, "early_stopping": True},
        hyperparameter_space={
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
            "learning_rate_init": [0.001, 0.01],
        },
        requires_scaling=True,
        supports_proba=True,
    ),
    "naive_bayes": ModelConfig(
        name="Gaussian Naive Bayes",
        model_class="sklearn.naive_bayes.GaussianNB",
        default_params={},
        hyperparameter_space={
            "var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
        },
        requires_scaling=False,
        supports_proba=True,
    ),
    "knn": ModelConfig(
        name="K-Nearest Neighbors",
        model_class="sklearn.neighbors.KNeighborsClassifier",
        default_params={"n_jobs": -1},
        hyperparameter_space={
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "p": [1, 2],
        },
        requires_scaling=True,
        supports_proba=True,
    ),
    "extra_trees": ModelConfig(
        name="Extra Trees",
        model_class="sklearn.ensemble.ExtraTreesClassifier",
        default_params={"random_state": 42, "n_jobs": -1},
        hyperparameter_space={
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
        requires_scaling=False,
        supports_proba=True,
    ),
}

# Cross-validation configuration
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Training configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_random_search_iter": 50,  # Number of iterations for RandomizedSearchCV
}

# Evaluation thresholds
EVALUATION_THRESHOLDS = {
    "min_accuracy": 0.55,  # Minimum acceptable accuracy (better than coin flip)
    "min_auc": 0.55,  # Minimum acceptable AUC
    "min_brier_improvement": 0.01,  # Minimum improvement over baseline
}
