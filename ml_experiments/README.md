# ML Experiments

This package provides comprehensive ML experimentation capabilities for NBA game win predictions, including model comparison, hyperparameter tuning, and evaluation utilities.

## Package Structure

| File                     | Description                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| `config.py`              | Feature definitions, model configurations, and hyperparameter search spaces for 10+ models |
| `data_generator.py`      | Synthetic data generation for training augmentation and testing                            |
| `feature_engineering.py` | Feature transformations, derived features, scaling, and feature selection                  |
| `models.py`              | Model factory with dynamic instantiation, voting/stacking ensembles                        |
| `evaluation.py`          | Comprehensive metrics (accuracy, ROC AUC, Brier score, calibration error)                  |
| `training_pipeline.py`   | End-to-end training with cross-validation and hyperparameter tuning                        |
| `model_comparison.py`    | Multi-model benchmarking, statistical significance tests, production readiness checks      |
| `run_experiments.py`     | CLI entry point for running experiments                                                    |

## Quick Start

### Run with Synthetic Data

```bash
# Quick comparison (~1-2 min)
python -m ml_experiments.run_experiments --synthetic --samples 5000

# With hyperparameter tuning (~5-10 min)
python -m ml_experiments.run_experiments --synthetic --samples 5000 --tune

# Save results to directory
python -m ml_experiments.run_experiments --synthetic --samples 10000 --tune --output results/
```

### Run with Real Data

```bash
# Basic comparison
python -m ml_experiments.run_experiments --data path/to/past_games.csv

# With hyperparameter tuning
python -m ml_experiments.run_experiments --data path/to/past_games.csv --tune

# With data augmentation (adds synthetic samples to real data)
python -m ml_experiments.run_experiments --data path/to/past_games.csv --augment

# Feature analysis only
python -m ml_experiments.run_experiments --data path/to/past_games.csv --analyze-features
```

### Programmatic Usage

```python
from ml_experiments import TrainingPipeline, ModelComparison, SyntheticDataGenerator

# Initialize pipeline
pipeline = TrainingPipeline(random_state=42, test_size=0.2, cv_folds=5)

# Load and prepare data
df = pipeline.load_data("past_games.csv")
X, y = pipeline.prepare_data(df, add_derived_features=True)
pipeline.split_data(X, y)

# Train all available models
results = pipeline.train_all_models(hyperparameter_tuning=True)

# Compare and get best model
comparison = ModelComparison()
result = comparison.compare_models(
    models={name: r.best_model for name, r in results.items()},
    X_train=pipeline.X_train,
    y_train=pipeline.y_train,
    X_test=pipeline.X_test,
    y_test=pipeline.y_test,
)

print(comparison.generate_comparison_report(result))
```

## Available Models

| Model                | Requires Scaling | Notes                             |
| -------------------- | ---------------- | --------------------------------- |
| Logistic Regression  | Yes              | Good baseline, interpretable      |
| Random Forest        | No               | Handles non-linear patterns       |
| Gradient Boosting    | No               | Often best performer              |
| Extra Trees          | No               | Fast, good for feature importance |
| XGBoost\*            | No               | State-of-the-art boosting         |
| LightGBM\*           | No               | Fast gradient boosting            |
| SVM                  | Yes              | Good with proper tuning           |
| MLP (Neural Network) | Yes              | Can capture complex patterns      |
| K-Nearest Neighbors  | Yes              | Simple, interpretable             |
| Naive Bayes          | No               | Fast baseline                     |

\*Requires optional dependencies: `uv sync --group ml-experiments`

## Understanding the Output

### Key Metrics

| Metric            | Range   | Goal   | Description                          |
| ----------------- | ------- | ------ | ------------------------------------ |
| ROC AUC           | 0.5-1.0 | Higher | Overall discriminative ability       |
| Accuracy          | 0-1.0   | Higher | Correct predictions / total          |
| Brier Score       | 0-1.0   | Lower  | Probability calibration quality      |
| Calibration Error | 0-1.0   | Lower  | How well probabilities match reality |
| CV Std            | 0+      | Lower  | Model stability across folds         |

### Realistic Targets for NBA Prediction

| Level                     | Accuracy | ROC AUC   |
| ------------------------- | -------- | --------- |
| Random                    | 50%      | 0.50      |
| Baseline (majority class) | ~55%     | 0.55      |
| Good Model                | 55-58%   | 0.58-0.62 |
| Excellent                 | 60%+     | 0.65+     |

## Next Steps: Features to Add

The biggest improvements come from better features, not fancier models. Consider adding these to your data pipeline:

### High Priority

```python
# Player availability (beyond is_top_players)
"home_total_war_available"      # Sum of WAR for active players
"away_injury_impact"            # WAR of injured/resting players
"star_player_matchup"           # When stars face each other

# Travel and fatigue
"home_travel_miles_last_week"   # Cumulative travel fatigue
"away_timezone_changes"         # Jet lag indicator
"is_altitude_game"              # Denver home games (5280 ft)

# Schedule factors
"home_games_in_last_7_days"     # Schedule density
"is_second_of_back_to_back"     # More specific than days_rest=0
"days_until_next_game"          # Teams rest starters if next game soon
```

### Medium Priority

```python
# Advanced team stats
"home_offensive_rating"         # Points per 100 possessions
"home_defensive_rating"         # Points allowed per 100 possessions
"pace_differential"             # Fast vs slow team matchups
"three_point_attempt_rate"      # Modern NBA style indicator
"free_throw_rate"               # Drawing fouls ability

# Matchup-specific
"h2h_record_last_5"             # Head-to-head history
"home_vs_conference_record"     # East vs West performance
"home_vs_division_record"       # Division rival intensity
```

### Lower Priority (But Potentially Valuable)

```python
# Market/betting data
"opening_line"                  # Where line opened
"line_movement"                 # Sharp money indicator
"total_over_under"              # Expected game pace
"public_betting_percentage"     # Fade the public signal

# Situational
"is_nationally_televised"       # Teams play differently on TV
"is_playoff_seeding_relevant"   # Late season motivation
"days_since_coaching_change"    # New coach bump/adjustment
"is_revenge_game"               # Playing former team
```

### Feature Engineering Tips

1. **Differentials matter more than absolutes** - `home_win_pct - away_win_pct` is more predictive than either alone
2. **Recency weighting** - Last 10 games often better than season averages
3. **Interaction terms** - "good team on back-to-back" behaves differently than components suggest
4. **Normalize by opponent** - Strength of schedule adjustments

## Deploying a New Model

Once you find a better model:

```python
from ml_experiments import TrainingPipeline

# Train on ALL historical data (no holdout)
pipeline = TrainingPipeline()
df = pipeline.load_data("full_historical_data.csv")
X, y = pipeline.prepare_data(df, add_derived_features=True)

# Fit the winning model type
best_model = pipeline.factory.create_model("gradient_boosting", params=best_params)
best_model.fit(X, y)

# Save to production location
pipeline.save_model(best_model, "src/log_model.joblib")
```

Then update `src/utils.py` if needed (e.g., if switching to a model that requires feature scaling).
