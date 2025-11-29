# ML Experiments (NBA Prediction V2)

This package provides a production-grade machine learning workflow for NBA win prediction. It is designed to prevent data leakage (Look-ahead bias), validate models using realistic Time-Series splits, and simulate financial performance (ROI) against betting odds.

## Package Structure

| File                     | Description                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------- |
| `config.py`              | V2 Schema definitions (VORP, Travel), model configs, and hyperparameter search spaces.  |
| `data_generator.py`      | Synthetic data generation using Latent Variable logic (correlating rank/wins/score).    |
| `feature_engineering.py` | Fatigue indexing, VORP differentials, skewness handling (Log1p), and feature selection. |
| `models.py`              | Model factory with automatic Pipeline wrapping (Scaling) and artifact persistence.      |
| `evaluation.py`          | Metrics (MCC, Brier, Calibration ECE) and Vectorized Betting Simulation.                |
| `training_pipeline.py`   | End-to-end orchestration enforcing Time-Series Splitting to prevent leakage.            |
| `model_comparison.py`    | Benchmarking, statistical significance tests, and Profit/ROI checks.                    |
| `run_experiments.py`     | CLI entry point for running experiments.                                                |

## Quick Start

### 1. Run with Synthetic Data

Great for testing the pipeline flow without needing a database connection.

```bash
# Generate 2000 samples and run full benchmark + financial sim (~30 secs)
python -m ml_experiments.run_experiments --synthetic --samples 2000
```

### 2\. Run with Real Data

Requires a CSV matching the V2 Schema (see `config.py`).

```bash
# Basic comparison (Train on past, Test on future)
python -m ml_experiments.run_experiments --data data/nba_games_v2.csv

# With Hyperparameter Tuning and Financial ROI Check
python -m ml_experiments.run_experiments --data data/nba_games_v2.csv --tune --financial
```

### 3\. Programmatic Usage

```python
from ml_experiments import TrainingPipeline, ModelComparison

# 1. Initialize Pipeline (Enforces Time-Series Split)
pipeline = TrainingPipeline(test_size=0.2, cv_splits=5)

# 2. Load & Prep (Splits data temporally to prevent leakage)
pipeline.load_and_prep_data("data/nba_games_v2.csv")

# 3. Train & Tune specific model
result = pipeline.train_model("xgboost", hyperparameter_tuning=True)

# 4. Check Financials (ROI)
print(f"Test AUC: {result.test_metrics.auc}")
print(f"ROI: {result.test_metrics.roi_simulation}%")

# 5. Save Artifacts (Model + Feature Engineer state)
pipeline.save_artifacts(result, "models/prod_model.joblib")
```

## V2 Feature Schema

The new model relies on "Situation & Talent" rather than just raw stats.

| Category | Key Features                                             | Rationale                                                                              |
| :------- | :------------------------------------------------------- | :------------------------------------------------------------------------------------- |
| Fatigue  | `home_fatigue_index`, `rest_diff`, `travel_miles_last_7` | NBA teams underperform significantly when tired or traveling cross-country.            |
| Talent   | `home_active_vorp`, `star_score`, `vorp_missing_pct`     | Value Over Replacement Player (VORP) tracks raw talent availability better than win %. |
| Momentum | `recent_form_diff`, `net_rating_last10`                  | Captures hot/cold streaks vs season averages.                                          |
| Context  | `is_cross_country_trip`, `home_advantage`                | Specific situational disadvantages.                                                    |

## Available Models

| Model                | Auto-Scaling | Use Case                                  |
| -------------------- | ------------ | ----------------------------------------- |
| Logistic Regression  | Yes          | Baseline, highly calibratable.            |
| Random Forest        | No           | Robust to noise, no tuning needed.        |
| XGBoost\*            | No           | Current SOTA. High accuracy.              |
| LightGBM\*           | No           | Faster training on large datasets.        |
| MLP (Neural Network) | Yes          | Captures non-linear fatigue interactions. |

_\*Requires optional dependencies: `uv sync --group ml-experiments`_

## Understanding the Output

### Key Metrics

| Metric            | Goal    | Description                                                  |
| ----------------- | ------- | ------------------------------------------------------------ |
| ROI (Kelly)       | \> 0%   | Return on Investment using fractional Kelly staking.         |
| MCC               | \> 0.15 | Matthews Correlation Coeff. Better than Accuracy for sports. |
| ROC AUC           | \> 0.60 | Ability to rank winners vs losers.                           |
| Calibration (ECE) | \< 0.05 | Critical for betting. Low ECE = Trustworthy Probabilities.   |

### Realistic Targets

| Level            | Win Rate | ROI         |
| ---------------- | -------- | ----------- |
| Random           | 50%      | -4.5% (Vig) |
| Public Model     | 62-64%   | -1.0%       |
| Profitable Model | 66%+     | +2.0%+      |

## Next Steps for Improvement

1.  Market Efficiency Features:
    - `closing_line_value`: Did the line move against us?
    - `public_percentage`: Are we betting with or against the crowd?
2.  Player-Level Granularity:
    - Replace `star_score` with aggregated `DARKO` or `EPM` projections for the specific active roster.
3.  Matchup Specifics:
    - `offensive_style` vs `defensive_style` (e.g., Fast Pace team vs Slow Pace team).

## Current v2 Model Benchmark (Synthetic Data)

Generated Nov 29, 2025 using `python -m ml_experiments.run_experiments --synthetic --samples 10000 --tune`.

- Because the V2 Feature Schema (Fatigue, VORP, Travel) is new and lacks historical point-in-time data, the current production model was trained on Synthetic Data to bootstrap the pipeline.

| Rank | Model               | Test AUC | Test Accuracy | Notes                   |
| :--- | :------------------ | :------- | :------------ | :---------------------- |
| 1    | Logistic Regression | 0.785    | 70.9%         | Selected for Production |
| 2    | MLP (Neural Net)    | 0.779    | 70.5%         |                         |
| 3    | Gradient Boosting   | 0.779    | 70.3%         |                         |
| 4    | Random Forest       | 0.772    | 70.9%         |                         |
| 5    | XGBoost             | 0.743    | 67.5%         | Overfitted to noise     |
