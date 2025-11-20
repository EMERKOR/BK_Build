# Changelog

All notable changes to the Ball Knower project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **v1.3 Model Calibration Pipeline**:
  - `ball_knower/evaluation/calibration_v1_3.py` - Calibration utilities for computing bias, MAE, RMSE, and edge-bin ATS rates
  - `src/calibrate_v1_3.py` - CLI script to generate calibration parameters from backtest results
  - `bk_build calibrate-v1-3` subcommand for streamlined calibration workflow
  - Unit tests for v1.3 calibration utilities (`tests/test_calibration_v1_3.py`)
- **v1.3 Backtest Integration**:
  - v1.3 support in `src/run_backtests.py` - loads trained v1.3 model and generates backtest results
  - v1.3 support in `bk_build backtest` CLI subcommand
  - Validation to reject v1.3 backtests before 2013 (EPA data availability constraint)
  - Smoke tests for v1.3 backtest CLI integration (`tests/test_backtest_cli.py`)
- Weekly prediction sanity validation module (`ball_knower/validation/sanity.py`)
- Diagnostic and fallback handling for missing weekly data sources
- v1.2 evaluation notebook (`notebooks/v1_2_evaluation.ipynb`)
- Placeholder team form feature module for v1.3 (`ball_knower/features/form.py`)
- Isolation tests to prevent accidental v1.3 feature usage in v1.2
- Model version banners for all major scripts
- CHANGELOG.md for version tracking

### Changed
- Enhanced `src/run_weekly_predictions.py` with data diagnostics and sanity checks
- Improved error handling and fallback logic for missing datasets
- Updated `docs/model_versions.json` with v1.3 calibration and backtest CLI usage

## [2.0.0] - 2025-11-20

### Added
- Centralized configuration in `ball_knower/config.py`
- Unified path helpers in `ball_knower/utils/paths.py`
- Data schema validation module (`ball_knower/io/schemas.py`)
- Schema validation integrated into all data loaders
- Comprehensive schema validation tests (`tests/test_schemas.py`)
- Unified CLI wrapper (`src/bk_build.py`) with subcommands:
  - `train-v1-2`: Train v1.2 model
  - `backtest`: Run backtests for v1.0/v1.2
  - `predict`: Generate weekly predictions
  - `export-predictiontracker`: Export to PredictionTracker format
- Machine-readable model metadata (`docs/model_versions.json`)
- Usage documentation (`docs/USAGE_BK_BUILD.md`)
- Version utilities for reproducibility banners

### Changed
- Deprecated `src/config.py` in favor of `ball_knower/config.py`
- Updated all imports across codebase to use centralized config
- Refactored scripts to use standardized path helpers
- Organized output directory structure (models/, backtests/, predictions/)

### Fixed
- Team name normalization issues in data loaders
- Schema validation warnings now handled gracefully

## [1.2.0] - 2024-XX-XX (v1.2 Consolidation Build)

### Added
- v1.2 model with ML correction layer
- Advanced structural features (rest, form, QB adjustments)
- v1.2 dataset builder (`ball_knower/datasets/v1_2.py`)
- v1.2 consistency tests
- PredictionTracker export functionality
- Evaluation module for model metrics

### Changed
- Improved feature engineering with leak-free rolling features
- Enhanced backtest infrastructure for multiple model versions

## [1.1.0] - 2024-XX-XX (v1.1 Enhanced Features)

### Added
- Enhanced spread model with structural features
- Calibrated weight loading from JSON
- Rest advantage calculations
- Win rate form metrics (L5 games)
- QB adjustment differentials
- Dynamic weight recalibration support

### Changed
- Expanded from deterministic baseline to feature-rich model
- Added EPA metrics beyond simple ratings

## [1.0.0] - 2024-XX-XX (v1.0 Deterministic Baseline)

### Added
- Deterministic baseline model using power rating differentials
- nfelo integration
- Substack ratings integration
- Basic home field advantage (HFA) calibration
- Initial data loading infrastructure
- Team name mapping and normalization
- Weekly prediction pipeline

### Changed
- Established spread prediction conventions (home team perspective)
- Negative spread = home favored
- Positive spread = home underdog

## Project Organization Changes

### Infrastructure Improvements
- Created `ball_knower/` package structure
- Separated features, datasets, and IO modules
- Added comprehensive test suite
- Established clear module boundaries
- Implemented schema validation for data integrity

### Documentation
- Added usage guide (`USAGE_BK_BUILD.md`)
- Created model version metadata (`model_versions.json`)
- Documented data setup procedures (`DATA_SETUP_GUIDE.md`)
- Added inline documentation for all modules

### Developer Experience
- Unified CLI for all operations
- Consistent path management
- Clear error messages with diagnostic information
- Version banners for reproducibility
- Automated sanity checks

---

## Legend

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security fixes

## Version Numbering

Ball Knower follows semantic versioning:
- **Major version**: Incompatible model changes or architecture overhauls
- **Minor version**: New features, backward-compatible
- **Patch version**: Bug fixes, minor improvements

## Notes

- v1.0, v1.1, v1.2 refer to model versions, not software versions
- Software version 2.0.0 represents the consolidation and infrastructure build
- All model math and feature definitions remain stable across infrastructure changes
