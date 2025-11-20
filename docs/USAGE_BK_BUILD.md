# Ball Knower Build System - Usage Guide

Internal reference for training, backtesting, and deploying Ball Knower models.

## Overview

The Ball Knower Build System (`bk_build.py`) provides a unified CLI for:

1. **Training**: Train v1.2 model on historical data
2. **Backtesting**: Evaluate model performance across seasons
3. **Predicting**: Generate weekly predictions for current season
4. **Exporting**: Export predictions to PredictionTracker format

## Installation & Setup

```bash
# Install package in editable mode (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements-dev.txt

# Verify the CLI is executable
python src/bk_build.py --help
```

## Quickstart for a Weekly Run

The fastest way to generate predictions for the current week:

**Step 1: Place data files in `data/current_season/`:**
- `power_ratings_nfelo_2025_week_11.csv`
- `epa_tiers_nfelo_2025_week_11.csv`
- `power_ratings_substack_2025_week_11.csv`
- `weekly_projections_ppg_substack_2025_week_11.csv`

**Step 2: Run the one-shot pipeline:**
```bash
python src/bk_build.py weekly-pipeline --week 11
```

**What it does:**
1. Validates all data files are present and correctly formatted (`check-weekly-data`)
2. Generates predictions using the v1.3 model (default)
3. Saves predictions to `output/predictions/v1_3/predictions_2025_week_11.csv`

**Customize the pipeline:**
```bash
# Use a different model version
python src/bk_build.py weekly-pipeline --week 11 --model-version v1.2

# Run for a different season
python src/bk_build.py weekly-pipeline --season 2024 --week 18

# Include backtest and export (not yet implemented)
python src/bk_build.py weekly-pipeline --week 11 --backtest --export-predictiontracker
```

**Validate data only (without predictions):**
```bash
python src/bk_build.py check-weekly-data --week 11
```

For detailed documentation of individual commands, see sections below.

## Model Versions

| Version | Description | Target | Use Case |
|---------|-------------|--------|----------|
| **v1.0** | Deterministic baseline (nfelo + HFA) | Actual margin | Baseline comparison |
| **v1.1** | Enhanced with structural features | Actual margin | Live weekly predictions |
| **v1.2** | ML correction layer | Vegas spread | Advanced backtesting |

See `docs/model_versions.json` for full model metadata.

## Training Models

### Train v1.2 Model

```bash
# Train v1.2 on historical data (2009-2024)
python src/bk_build.py train-v1-2 --start-season 2009 --end-season 2024

# Train on custom season range
python src/bk_build.py train-v1-2 --start-season 2015 --end-season 2023
```

**Note**: Training v1.2 requires nfelo historical data. Model artifacts are saved to `output/models/v1_2/`.

### Calibrate v1.1 Weights

```bash
# Calibrate v1.1 weights and export to JSON
python calibrate_v1_json.py

# Weights saved to: output/calibrated_weights_v1.json
# These weights are automatically loaded by v1.1 model
```

## Backtesting

### Backtest v1.0 (Baseline)

```bash
# Backtest v1.0 across 2019-2024 seasons
python src/bk_build.py backtest --model v1.0 \
    --start-season 2019 --end-season 2024

# With edge threshold (only bet when |edge| >= 0.5)
python src/bk_build.py backtest --model v1.0 \
    --start-season 2019 --end-season 2024 \
    --edge-threshold 0.5

# Custom output path
python src/bk_build.py backtest --model v1.0 \
    --start-season 2019 --end-season 2024 \
    --output my_backtests/v1_0_results.csv
```

**Default output**: `output/backtests/v1_0/backtest_2019_2024.csv`

### Backtest v1.2 (ML Model)

```bash
# Backtest v1.2 across 2019-2024 seasons
python src/bk_build.py backtest --model v1.2 \
    --start-season 2019 --end-season 2024

# With edge threshold
python src/bk_build.py backtest --model v1.2 \
    --start-season 2019 --end-season 2024 \
    --edge-threshold 1.0
```

**Default output**: `output/backtests/v1_2/backtest_2019_2024.csv`

**Requirements**:
- v1.2 model must be trained first
- Model file: `output/ball_knower_v1_2_model.json`

## Weekly Data Validation

Before running weekly predictions, validate that all required data files are present and valid:

```bash
# Check weekly data for 2025 Week 11
python src/bk_build.py check-weekly-data --week 11

# Check data for a different season
python src/bk_build.py check-weekly-data --season 2024 --week 18
```

**Output example**:
```
[Weekly Data Check] Season 2025, Week 11
================================================================================

Required nfelo files:
  ✓ power_ratings_nfelo_2025_week_11.csv - Valid (32 rows)
  ✓ epa_tiers_nfelo_2025_week_11.csv - Valid (32 rows)
  ⚠ strength_of_schedule_nfelo_2025_week_11.csv - Optional file not found

Required Substack files:
  ✓ power_ratings_substack_2025_week_11.csv - Valid (32 rows)
  ⚠ qb_epa_substack_2025_week_11.csv - Optional file not found
  ✓ weekly_projections_ppg_substack_2025_week_11.csv - Valid (16 rows)

================================================================================
Summary:
  Required files: 4/4 ✓
  Optional files: 0/2 ✓

✓ All required files present and valid
  Ready to run weekly predictions!
================================================================================
```

**Exit codes**:
- `0`: All required files present and valid
- `1`: Missing or invalid required files

## One-Shot Weekly Pipeline

Run the complete weekly workflow with a single command:

```bash
# Run full workflow for Week 11 with v1.3 model
python src/bk_build.py weekly-pipeline \
  --season 2025 \
  --week 11 \
  --model-version v1.3 \
  --backtest \
  --export-predictiontracker

# Minimal workflow (data check + predictions only)
python src/bk_build.py weekly-pipeline --week 11

# Use different model
python src/bk_build.py weekly-pipeline --week 12 --model-version v1.2
```

**What it does**:
1. Validates all required data files (same as `check-weekly-data`)
2. Generates predictions using specified model
3. Optionally runs backtest for this week (if `--backtest` flag provided)
4. Optionally exports to PredictionTracker format (if `--export-predictiontracker` flag provided)

**Default behavior**:
- Season: Current season from `config.CURRENT_SEASON`
- Model: v1.3 (most recent)
- Backtest: Disabled
- PredictionTracker export: Disabled

**Note**: Backtest and PredictionTracker export features are not yet implemented.

## Weekly Predictions

### Generate Predictions for Current Week

```bash
# Generate v1.1 predictions for 2025 Week 11
python src/bk_build.py predict --model v1.1 \
    --season 2025 --week 11

# Using v1.0 model
python src/bk_build.py predict --model v1.0 \
    --season 2025 --week 11

# Custom output path
python src/bk_build.py predict --model v1.1 \
    --season 2025 --week 12 \
    --output predictions_week12.csv
```

**Default output**: `output/predictions_{season}_week_{week}.csv`

**Requirements**:
- Current week data files in `data/current_season/`:
  - `power_ratings_nfelo_{season}_week_{week}.csv`
  - `power_ratings_substack_{season}_week_{week}.csv`
  - `epa_tiers_nfelo_{season}_week_{week}.csv`
  - `qb_epa_substack_{season}_week_{week}.csv`
  - `weekly_projections_ppg_substack_{season}_week_{week}.csv`

## PredictionTracker Export

```bash
# Export v1.2 predictions to PredictionTracker format
python src/bk_build.py export-predictiontracker \
    --model v1.2 --start-season 2019 --end-season 2024

# Custom output path
python src/bk_build.py export-predictiontracker \
    --model v1.2 --start-season 2019 --end-season 2024 \
    --output exports/predictiontracker_v1_2.csv
```

**Default output**: `output/predictiontracker/{model}_{start}_{end}.csv`

**Note**: This functionality is not yet fully implemented in the CLI.

## Output Directory Structure

The Ball Knower build system uses centralized path helpers (`ball_knower.utils.paths`) to organize outputs:

```
output/
├── models/
│   ├── v1_0/
│   │   └── (model artifacts)
│   └── v1_2/
│       └── ball_knower_v1_2_model.json
├── backtests/
│   ├── v1_0/
│   │   └── backtest_2019_2024.csv
│   └── v1_2/
│       └── backtest_2019_2024.csv
├── predictions/
│   └── predictions_2025_week_11.csv
├── predictiontracker/
│   └── v1_2_2019_2024.csv
└── calibrated_weights_v1.json
```

## Running Tests

### Run All Tests

```bash
# Run full test suite with pytest
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ball_knower --cov=src -v
```

### Run Specific Test Suites

```bash
# Test v1.2 consistency
pytest tests/test_v1_2_consistency.py -v

# Test schema validation
pytest tests/test_schemas.py -v

# Test data loaders
pytest tests/test_loaders.py -v

# Test models
pytest tests/test_models.py -v
```

### Key Tests to Run Before Deployment

1. **v1.2 Consistency**: Ensures v1.2 predictions are deterministic
   ```bash
   pytest tests/test_v1_2_consistency.py -v
   ```

2. **Schema Validation**: Ensures data integrity checks work
   ```bash
   pytest tests/test_schemas.py -v
   ```

3. **Loader Tests**: Ensures data files load correctly
   ```bash
   pytest tests/test_loaders.py -v
   ```

## Troubleshooting

### Model File Not Found

**Error**: `v1.2 model file not found at output/ball_knower_v1_2_model.json`

**Solution**: Train the v1.2 model first:
```bash
python src/bk_build.py train-v1-2 --start-season 2009 --end-season 2024
```

### Data Files Not Found

**Error**: `Could not find file for category='power_ratings', provider='nfelo'`

**Solution**: Ensure current week data files exist in `data/current_season/` with correct naming:
- Category-first: `power_ratings_nfelo_2025_week_11.csv`
- Provider-first (legacy): `nfelo_power_ratings_2025_week_11.csv`

### Schema Validation Errors

**Error**: `Missing required columns in nfelo power ratings: ['nfelo']`

**Solution**: Check that your data files have the correct column names. Use schema validators for debugging:
```python
from ball_knower.io import loaders, schemas

df = loaders.load_power_ratings("nfelo", season=2025, week=11)
schemas.validate_nfelo_power_ratings_df(df)
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'ball_knower'`

**Solution**: Ensure project root is in your PYTHONPATH:
```bash
export PYTHONPATH=/path/to/BK_Build:$PYTHONPATH
```

Or run scripts from project root:
```bash
cd /path/to/BK_Build
python src/bk_build.py --help
```

## Configuration

### Global Configuration

All global constants are centralized in `ball_knower/config.py`:

- `HOME_FIELD_ADVANTAGE = 2.5` points
- `TRAINING_START_YEAR = 2015`
- `TRAINING_END_YEAR = 2024`
- `CURRENT_SEASON = 2025`
- `CURRENT_WEEK = 11`

To modify configuration:
```python
from ball_knower import config

# View current config
print(config.get_config_summary())

# Modify for Colab environment
config.setup_colab_paths()
```

### Path Configuration

All output paths are managed through `ball_knower.utils.paths`:

```python
from ball_knower.utils import paths

# Get path helpers
output_dir = paths.get_output_dir()
models_dir = paths.get_models_dir("v1.2")
backtest_path = paths.get_backtest_results_path("v1.2", 2019, 2024)
```

## Advanced Usage

### Using Models Programmatically

```python
from src import models
from ball_knower import config

# Load v1.1 with calibrated weights
model = models.EnhancedSpreadModel(use_calibrated=True)

# Make prediction
home_features = {'nfelo': 1650, 'epa_margin': 0.15, ...}
away_features = {'nfelo': 1600, 'epa_margin': 0.10, ...}
spread = model.predict(home_features, away_features)

print(f"Predicted spread: {spread:.1f}")
```

### Loading Data Programmatically

```python
from ball_knower.io import loaders

# Load all current week data
data = loaders.load_all_sources(season=2025, week=11)

# Access individual sources
nfelo_power = data['power_ratings_nfelo']
substack_power = data['power_ratings_substack']
merged = data['merged_ratings']

print(f"Loaded {len(merged)} teams")
```

## Additional Resources

- **Model Metadata**: `docs/model_versions.json`
- **Data Setup Guide**: `DATA_SETUP_GUIDE.md`
- **Repository README**: `README.md`
- **Feature Engineering**: `ball_knower/features/engineering.py`
- **Schema Definitions**: `ball_knower/io/schemas.py`

## Support

For issues or questions:
1. Check this usage guide
2. Review test files for examples
3. Consult model documentation in `docs/model_versions.json`
4. Open an issue on GitHub: `EMERKOR/BK_Build`
