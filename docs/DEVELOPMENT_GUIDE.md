# Development Guide

[Docs Home](README.md) | [Architecture](ARCHITECTURE_OVERVIEW.md) | [Data Sources](DATA_SOURCES.md) | [Feature Tiers](FEATURE_TIERS.md) | [Spec](BALL_KNOWER_SPEC.md) | [Dev Guide](DEVELOPMENT_GUIDE.md)

---

## Overview

This guide provides best practices, guidelines, and procedures for contributing to Ball Knower. It covers running tests, adding new components, and safe migration practices.

---

## Running Tests

### Fixture-Based Testing

All Ball Knower tests use **frozen fixtures** to ensure reproducibility and prevent regression.

**Why Fixture-Based?**
- **Deterministic**: Same inputs always produce same outputs
- **Fast**: No network calls or large data downloads
- **Reliable**: Tests don't break when external data changes
- **Isolated**: Each test is independent

### Running the Test Suite

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_loaders.py

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=ball_knower
```

### Test Organization

```
tests/
├── fixtures/              # Frozen test data
│   ├── power_ratings_nfelo_sample.csv
│   ├── qb_metrics_sample.csv
│   └── games_sample.csv
├── test_loaders.py        # Data loader tests
├── test_features.py       # Feature computation tests
└── test_models.py         # Model training tests
```

### Writing New Tests

**Template**:
```python
import pytest
import pandas as pd
from ball_knower.io.loaders import load_power_ratings

def test_load_power_ratings():
    """Test that power ratings loader works correctly."""
    # Load fixture
    df = pd.read_csv("tests/fixtures/power_ratings_nfelo_sample.csv")

    # Validate schema
    assert "team" in df.columns
    assert "elo_rating" in df.columns

    # Validate data
    assert len(df) > 0
    assert df["elo_rating"].min() > 0
```

---

## Adding New Loaders

### Checklist

When adding a new data loader:

1. **Create loader module** in `ball_knower/io/loaders/`
2. **Follow naming convention** (category-first)
3. **Normalize team names** to standard abbreviations
4. **Validate required columns** exist
5. **Handle missing values** gracefully
6. **Add unit tests** with fixtures
7. **Document in DATA_SOURCES.md**

### Loader Template

```python
# ball_knower/io/loaders/new_provider.py

import pandas as pd
from pathlib import Path
from ball_knower.config.paths import DATA_DIR

def load_new_data(category, season, week):
    """Load data from new provider.

    Args:
        category (str): Data category (e.g., 'power_ratings')
        season (int): NFL season year
        week (int): Week number

    Returns:
        pd.DataFrame: Normalized data with standard columns
    """
    # Try category-first
    path = DATA_DIR / category / "new_provider" / f"{category}_new_provider_{season}_week_{week}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Load and validate
    df = pd.read_csv(path)
    _validate_schema(df, required_columns=["team", "rating"])
    _normalize_team_names(df)

    return df

def _validate_schema(df, required_columns):
    """Validate that required columns exist."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _normalize_team_names(df):
    """Normalize team names to standard abbreviations."""
    # Implementation here
    pass
```

---

## Adding New Model Versions

### Checklist

When adding a new model version (e.g., v2.1):

1. **Create dataset builder** in `ball_knower/datasets/dataset_builder_v2_1.py`
2. **Create training module** in `ball_knower/modeling/train_v2_1.py`
3. **Define feature tiers** used by the model
4. **Add validation** for leakage prevention
5. **Create fixtures** for testing
6. **Update BALL_KNOWER_SPEC.md** with model details
7. **Integrate with backtest infrastructure**

### Dataset Builder Template

```python
# ball_knower/datasets/dataset_builder_v2_1.py

import pandas as pd
from ball_knower.io import loaders
from ball_knower.features import structural, core

class DatasetBuilderV21:
    """Build dataset for v2.1 model."""

    def __init__(self, season_start, season_end):
        self.season_start = season_start
        self.season_end = season_end

    def build(self):
        """Build complete dataset."""
        # Load raw data
        games = loaders.load_games(self.season_start, self.season_end)
        ratings = loaders.load_ratings(self.season_start, self.season_end)

        # Compute features
        df = self._compute_features(games, ratings)

        # Validate tiers
        self._validate_tiers(df)

        return df

    def _compute_features(self, games, ratings):
        """Compute all features for v2.1."""
        df = games.copy()

        # T0 features
        df['rest_days'] = structural.compute_rest_days(df)
        df['is_division_game'] = structural.compute_division_flag(df)

        # T1 features
        df['elo_diff'] = core.compute_elo_diff(df, ratings)

        return df

    def _validate_tiers(self, df):
        """Validate that only allowed tiers are present."""
        # Validation logic here
        pass
```

---

## Adding New Features

### Guidelines

**Before Adding a Feature**:

1. **Determine the tier** (T0, T1, T2, T3, or TX)
2. **Verify no leakage** (can you know this before kickoff?)
3. **Check for duplication** (does this feature already exist?)
4. **Document in FEATURE_TIERS.md**

### Feature Implementation Template

```python
# ball_knower/features/core.py

def compute_new_feature(df, window=5):
    """Compute new feature with rolling window.

    Args:
        df (pd.DataFrame): Game data sorted by team and date
        window (int): Rolling window size

    Returns:
        pd.Series: New feature values
    """
    # CRITICAL: Use .shift(1) to exclude current game
    return df.groupby('team')['raw_metric'].shift(1).rolling(window).mean()
```

### Leakage Prevention Checklist

When adding a new feature:

- [ ] Feature does NOT include current game statistics
- [ ] Feature uses `.shift(1)` for rolling calculations
- [ ] Feature is available before kickoff of prediction target
- [ ] Feature does NOT reference future weeks or games
- [ ] Feature has been manually inspected for temporal validity

### When to Use Leakage Checker

[Placeholder for leakage checker implementation]

- How to run automated leakage detection
- How to interpret leakage checker results
- Common leakage patterns to avoid

---

## Preventing Duplication

### Check Before Creating

Before adding new code:

1. **Search existing modules** for similar functionality
2. **Check feature definitions** in `ball_knower/features/`
3. **Review loaders** in `ball_knower/io/loaders/`
4. **Ask**: "Does this already exist in a slightly different form?"

### Refactoring Over Duplication

If you find duplicated code:

1. **Extract to utility function** in `ball_knower/utils/`
2. **Update all callers** to use the shared function
3. **Add tests** for the refactored function
4. **Document** in code comments

---

## Safe Migration Practices

### Working with Existing Code

When modifying existing modules:

1. **Run tests before changes** to establish baseline
2. **Make incremental changes** (not wholesale rewrites)
3. **Run tests after each change** to catch regressions
4. **Use fixtures** to validate output matches expected format

### Migration Checklist

- [ ] Identify all code that depends on the module being migrated
- [ ] Create fixtures for existing behavior
- [ ] Implement new version alongside old (if possible)
- [ ] Validate both versions produce identical output
- [ ] Switch callers to new version one at a time
- [ ] Remove old version only after all callers are migrated

### Rollback Strategy

If migration causes issues:

1. **Revert to previous version** immediately
2. **Document the issue** in detail
3. **Fix in isolation** with tests
4. **Re-attempt migration** after validation

---

## Release Process

### Placeholder: Versioning Strategy

[To be documented]

- How model versions are numbered (v1.0, v1.2, v2.0, etc.)
- How to tag releases in git
- How to archive old model versions

### Placeholder: Deployment Checklist

[To be documented]

- Pre-deployment validation steps
- How to deploy models to production
- How to monitor model performance
- How to roll back if needed

---

## Code Style Guidelines

### Python Conventions

- **PEP 8**: Follow standard Python style guide
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings for all public functions
- **Line length**: Max 100 characters (not strict 80)

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_SNAKE_CASE`

### Example

```python
from typing import Optional
import pandas as pd

DEFAULT_WINDOW_SIZE = 5

class DatasetBuilder:
    """Build datasets for model training."""

    def __init__(self, season: int):
        """Initialize dataset builder.

        Args:
            season: NFL season year
        """
        self.season = season

    def build(self, window: Optional[int] = None) -> pd.DataFrame:
        """Build the dataset.

        Args:
            window: Rolling window size (uses default if None)

        Returns:
            DataFrame with computed features
        """
        window = window or DEFAULT_WINDOW_SIZE
        # Implementation here
        pass
```

---

## Getting Help

### Resources

- **Documentation**: Read all docs in `docs/`
- **Tests**: Check `tests/` for usage examples
- **Code Examples**: Review existing modules in `ball_knower/`

### Common Questions

**Q: How do I add a new data source?**
A: See "Adding New Loaders" section above and [DATA_SOURCES.md](DATA_SOURCES.md)

**Q: How do I know if my feature has leakage?**
A: Ask "Could I know this value before kickoff?" and use `.shift(1)` for rolling stats

**Q: Where should I put my code?**
A: See [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) for module organization

---

## References

- [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) — System architecture and module organization
- [FEATURE_TIERS.md](FEATURE_TIERS.md) — Feature tier system and leakage prevention
- [DATA_SOURCES.md](DATA_SOURCES.md) — Data sources and loader documentation
- [BALL_KNOWER_SPEC.md](BALL_KNOWER_SPEC.md) — Model versions and specifications

---

**Status**: This guide is a living document and will be updated as development practices evolve.
