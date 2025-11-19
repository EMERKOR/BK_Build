# Architecture Overview

[Docs Home](README.md) | [Architecture](ARCHITECTURE_OVERVIEW.md) | [Data Sources](DATA_SOURCES.md) | [Feature Tiers](FEATURE_TIERS.md) | [Spec](BALL_KNOWER_SPEC.md) | [Dev Guide](DEVELOPMENT_GUIDE.md)

---

## Overview

This document describes the Ball Knower system architecture, including the canonical project layout, module organization, and design principles.

---

## Project Layout Diagram

```
BK_Build/
├── ball_knower/              # Canonical source of all production code
│   ├── config/               # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py       # Global settings and constants
│   │   └── paths.py          # File path management
│   │
│   ├── features/             # Feature definitions organized by tier
│   │   ├── __init__.py
│   │   ├── structural.py     # Tier 0 features
│   │   ├── core.py           # Tier 1 features
│   │   ├── market.py         # Tier 2 features
│   │   └── experimental.py   # Tier 3 features
│   │
│   ├── modeling/             # Model implementations and training logic
│   │   ├── __init__.py
│   │   ├── train_v1_0.py     # v1.0 training logic
│   │   ├── train_v1_2.py     # v1.2 training logic
│   │   ├── train_v1_3.py     # v1.3 training logic
│   │   └── train_v2_0.py     # v2.0 training logic
│   │
│   ├── io/                   # Input/output and data loading
│   │   ├── __init__.py
│   │   └── loaders/          # Data loaders for each provider
│   │       ├── __init__.py
│   │       ├── nfelo.py      # nfelo data loader
│   │       ├── substack.py   # Substack data loader
│   │       └── nflverse.py   # nflverse data loader
│   │
│   ├── datasets/             # Dataset builders for each model version
│   │   ├── __init__.py
│   │   ├── dataset_builder_v1_0.py
│   │   ├── dataset_builder_v1_2.py
│   │   ├── dataset_builder_v1_3.py
│   │   └── dataset_builder_v2_0.py
│   │
│   └── utils/                # Shared utilities
│       ├── __init__.py
│       ├── validation.py     # Data validation helpers
│       └── preprocessing.py  # Data preprocessing utilities
│
├── src/                      # CLI scripts and analysis tools
│   ├── backtest_unified.py   # Unified backtest driver
│   ├── predict_current_week.py  # Weekly prediction tool
│   └── analyze_results.py    # Results analysis script
│
├── data/                     # Raw and processed data
│   ├── elo/                  # Elo ratings (category-first)
│   ├── odds/                 # Betting odds
│   ├── stats/                # Team statistics
│   ├── historical/           # Historical datasets
│   └── reference/            # Reference data (team names, etc.)
│
├── tests/                    # Fixture-based unit tests
│   ├── fixtures/             # Test fixtures
│   ├── test_loaders.py       # Loader tests
│   ├── test_features.py      # Feature tests
│   └── test_models.py        # Model tests
│
├── docs/                     # Documentation (you are here!)
│   ├── README.md
│   ├── BALL_KNOWER_SPEC.md
│   ├── DATA_SOURCES.md
│   ├── FEATURE_TIERS.md
│   ├── ARCHITECTURE_OVERVIEW.md
│   └── DEVELOPMENT_GUIDE.md
│
├── notebooks/                # Jupyter notebooks for exploration
│   └── exploratory/          # Exploratory analysis notebooks
│
└── output/                   # Model outputs and predictions
    ├── predictions/          # Weekly predictions
    ├── backtest_results/     # Backtest outputs
    └── models/               # Trained model artifacts
```

---

## Canonical Structure

### ball_knower/ — Core Package

This is the **canonical source** of all production code. All production logic, feature definitions, model training, and data loading must live here.

#### Why This Structure?

1. **Separation of Concerns**: Each module has a clear, single responsibility
2. **Discoverability**: Easy to find where specific functionality lives
3. **Testability**: Each module can be tested independently
4. **Extensibility**: Easy to add new features, models, or data sources

---

## Module Descriptions

### ball_knower/config/

**Purpose**: Centralized configuration management

**Contains**:
- `settings.py` — Global settings and constants (e.g., home field advantage, default parameters)
- `paths.py` — File path management for data, models, and outputs

**Example**:
```python
# settings.py
HOME_FIELD_ADVANTAGE = 2.5
DEFAULT_WINDOW_SIZES = [3, 5, 10]

# paths.py
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent.parent / "output" / "models"
```

---

### ball_knower/features/

**Purpose**: Feature engineering organized by tier

**Contains**:
- `structural.py` — Tier 0 structural features (always safe)
- `core.py` — Tier 1 core model features (primary predictors)
- `market.py` — Tier 2 market features (use with caution)
- `experimental.py` — Tier 3 experimental features (do not rely on)

**Design Principle**: Each file contains functions to compute features for that tier only.

**Example**:
```python
# structural.py
def compute_rest_days(df):
    """Compute days since last game for each team."""
    return df.groupby('team')['gameday'].diff().dt.days.fillna(7)

# core.py
def compute_elo_diff(df, ratings):
    """Compute nfelo rating differential (home - away)."""
    # Implementation here
```

---

### ball_knower/modeling/

**Purpose**: Model training logic for each version

**Contains**:
- `train_v1_0.py` — v1.0 deterministic baseline training
- `train_v1_2.py` — v1.2 residual model training
- `train_v1_3.py` — v1.3 score prediction training
- `train_v2_0.py` — v2.0 meta-model training

**Design Principle**: Each file is self-contained and handles training for one model version.

---

### ball_knower/io/loaders/

**Purpose**: Data loading and normalization

**Contains**:
- `nfelo.py` — Load nfelo power ratings and historical data
- `substack.py` — Load Substack power ratings and QB metrics
- `nflverse.py` — Load nflverse schedules and statistics

**Design Principle**: Each loader handles one data provider and returns normalized dataframes.

**Example**:
```python
# loaders/nfelo.py
def load_power_ratings(season, week):
    """Load nfelo power ratings for a specific week."""
    path = f"data/elo/nfelo/power_ratings_nfelo_{season}_week_{week}.csv"
    return pd.read_csv(path)
```

---

### ball_knower/datasets/

**Purpose**: Dataset construction for each model version

**Contains**:
- `dataset_builder_v1_0.py` — Build v1.0 dataset
- `dataset_builder_v1_2.py` — Build v1.2 dataset
- `dataset_builder_v1_3.py` — Build v1.3 dataset
- `dataset_builder_v2_0.py` — Build v2.0 dataset

**Design Principle**: Each builder orchestrates loading data, computing features, and validating tiers.

---

### src/ — CLI Scripts

**Purpose**: Command-line tools for running predictions and analysis

**Contains**:
- `backtest_unified.py` — Unified backtest driver for all model versions
- `predict_current_week.py` — Generate predictions for the current week
- `analyze_results.py` — Analyze backtest results and performance

**Design Principle**: Scripts are thin wrappers that call `ball_knower/` modules.

---

## Why This Architecture Exists

### 1. Prevent Duplication

By centralizing code in `ball_knower/`, we avoid duplicating logic across scripts, notebooks, and tests.

### 2. Enforce Tier System

The `ball_knower/features/` module structure makes it clear which features belong to which tier, preventing accidental leakage.

### 3. Enable Testing

All core logic in `ball_knower/` can be tested with fixtures, ensuring reproducibility and correctness.

### 4. Support Extensibility

Adding new features, models, or data sources follows a clear pattern:
- New feature tier? Add to `ball_knower/features/`
- New model version? Add to `ball_knower/modeling/` and `ball_knower/datasets/`
- New data provider? Add to `ball_knower/io/loaders/`

---

## How to Extend

### Placeholder: Adding New Features

[To be documented in DEVELOPMENT_GUIDE.md]

- How to add a new feature to an existing tier
- How to create a new feature tier
- How to validate features for leakage

### Placeholder: Adding New Model Versions

[To be documented in DEVELOPMENT_GUIDE.md]

- How to add a new model version (e.g., v2.1)
- How to create a new dataset builder
- How to integrate with backtest infrastructure

### Placeholder: Adding New Data Sources

[To be documented in DEVELOPMENT_GUIDE.md]

- How to add a new data provider
- How to create a new loader
- How to integrate with existing features

---

## Design Principles

### 1. Canonical Source of Truth

All production code lives in `ball_knower/`. Scripts in `src/` are thin wrappers.

### 2. Single Responsibility

Each module has one clear purpose (loaders load, features compute features, models train models).

### 3. Tier-Based Organization

Features are organized by tier to prevent leakage and make validation easier.

### 4. Testability First

All core logic is designed to be testable with frozen fixtures.

### 5. Extensibility by Default

Adding new components follows clear, documented patterns.

---

## References

- [BALL_KNOWER_SPEC.md](BALL_KNOWER_SPEC.md) — Model versions and workflow
- [FEATURE_TIERS.md](FEATURE_TIERS.md) — Feature organization and leakage prevention
- [DATA_SOURCES.md](DATA_SOURCES.md) — Data loading and schemas
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) — Extension guidelines

---

**Status**: This architecture is the foundation for all Ball Knower development.
