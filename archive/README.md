# Archived Scripts

This directory contains legacy scripts that have been superseded by the unified CLI tools in the main repository.

## Superseded By

These archived scripts are no longer actively maintained. Use the following official entrypoints instead:

- **Weekly Predictions**: Use `python src/run_weekly_predictions.py`
  - Supersedes: `run_demo.py`, `predict_current_week.py`

- **Backtests**: Use `python src/run_backtests.py`
  - Supersedes: `backtest_v1_0.py`, `backtest_v1_2.py`

- **Calibration**: Use `python calibrate_v1_json.py`
  - Supersedes: `calibrate_model.py`, `calibrate_to_vegas.py`, `calibrate_regression.py`, `calibrate_simple.py`

## Archived Script Categories

### Model Definitions (Legacy)
- `ball_knower_v1_1.py`
- `ball_knower_v1_2.py`
- `ball_knower_v1_final.py`
- `bk_v1_1_with_adjustments.py`
- `bk_v1_final.py`

Model logic has been consolidated into `src/models.py`.

### Investigation/Exploration Scripts
- `investigate_data.py`
- `explore_nflreadpy.py`
- `explore_nflverse_data.py`
- `enhanced_analysis_demo.py`
- `analyze_nfelo_features.py`

One-off exploratory analysis scripts used during initial development.

### Demo Scripts
- `run_demo.py`
- `predict_current_week.py`

Early demonstration scripts superseded by the comprehensive CLI tools.

### Backtest Scripts
- `backtest_v1_0.py`
- `backtest_v1_2.py`

Individual model backtests superseded by the unified backtest driver.

### Calibration Scripts
- `calibrate_model.py`
- `calibrate_to_vegas.py`
- `calibrate_regression.py`
- `calibrate_simple.py`

Various calibration approaches consolidated into `calibrate_v1_json.py`.

## Note

These scripts are preserved for reference but are not maintained. They may have outdated dependencies or data paths.
