# Ball Knower - NFL Betting Analytics

A leak-free, modular NFL spread prediction system that combines EPA analysis, power ratings, and machine learning to find value in betting markets.

## Project Goals

Build a reliable NFL spread prediction system that:
- Produces deterministic baseline spreads (v1.0)
- Enhances with structural features (v1.1)
- Applies ML correction layer (v1.2)
- Identifies value bets vs Vegas lines
- Provides ROI analysis for bet sizing

## Current Status

**COMPLETE - Core System Built**

- Team name normalization (handles all data sources)
- Data loaders for nfelo, Substack, and nfl_data_py
- Leak-free feature engineering framework
- v1.0 deterministic spread model
- v1.2 ML correction layer
- Backtest and ROI analysis functions
- Demo notebook with Week 11, 2025 predictions

## Architecture at a Glance

Ball Knower is organized into three main components:

### 1. Unified Data Loaders

**Module**: `ball_knower.io.loaders`

**Purpose**: Load current-week data from multiple providers with automatic team name normalization.

**Example**:
```python
from ball_knower.io import loaders

# Load all current-week sources
data = loaders.load_all_sources(season=2025, week=11)
power_ratings = data['power_ratings']  # Combined nfelo + Substack
qb_metrics = data['qb_metrics']        # QB EPA and performance
```

**File Convention**: Category-first naming (e.g., `power_ratings_nfelo_2025_week_11.csv`)

### 2. Historical Dataset Builders

Ball Knower has two dataset builders for different prediction tasks:

#### v1.0 - Actual Margin Prediction (Baseline Football Brain)

**Target**: Predicts actual game margin (`home_score - away_score`)

**Purpose**: Establishes the "football truth" - what should happen based purely on team strength

**Example**:
```python
from ball_knower.datasets import v1_0

# Build training dataset (2009-2024)
df_v1_0 = v1_0.build_training_frame()

# Features: nfelo ratings, EPA, power ratings, structural context
# Target: actual_margin
# Model: Linear regression baseline
```

**Use Case**: Foundation model that ignores betting markets

#### v1.2 - Vegas Spread Prediction (Market-Aware Model)

**Target**: Predicts Vegas closing spread (or learns spread corrections)

**Purpose**: Align with market consensus to identify genuine edges

**Example**:
```python
from ball_knower.datasets import v1_2

# Build training dataset with QB metrics and structural features
df_v1_2 = v1_2.build_training_frame()

# Features: All v1.0 features + QB metrics + seasonal context
# Target: vegas_spread_close
# Model: Ridge regression with regularization
```

**Use Case**: Compare model predictions to Vegas lines to find betting value

### 3. Model Progression

- **v1.0**: Pure football model (no market awareness)
- **v1.2**: Market-calibrated model (learns Vegas patterns)
- **Edge Detection**: `edge = v1_2_prediction - vegas_line`

### Core Principles

1. **Leak-Free Guarantee**: All features use `.shift(1)` on rolling stats
2. **Time-Based Splits**: Train on 2009-2024, test on 2025
3. **Interpretable**: Linear models with clear feature importance
4. **Modular**: Each version is self-contained and testable

## Quick Start

### Run the Demo Notebook

```bash
# Install dependencies
pip install pandas numpy scikit-learn nfl_data_py jupyter

# Start Jupyter
jupyter notebook notebooks/ball_knower_demo.ipynb
```

The demo notebook will:
1. Load your Week 11, 2025 data (nfelo + Substack)
2. Merge all team ratings
3. Generate spread predictions with v1.0 model
4. Identify value bets vs Vegas lines

### Run Weekly Predictions CLI

```bash
# Generate predictions for a specific week
python run_weekly_predictions.py --season 2025 --week 11

# Custom output location
python run_weekly_predictions.py --season 2025 --week 12 --output my_predictions.csv

# Use v1.0 model instead of default v1.1
python run_weekly_predictions.py --season 2025 --week 11 --model v1.0
```

**What it does**:
1. Loads all data sources for the specified week (nfelo, Substack, EPA, QB metrics)
2. Merges team ratings and builds feature matrix
3. Generates predictions using calibrated model weights (from `output/calibrated_weights_v1.json`)
4. Calculates edge vs Vegas lines
5. Saves predictions to CSV in `output/predictions_{season}_week_{week}.csv`

**Output CSV columns**:
- `game_id` - Unique identifier (e.g., `2025_11_BUF_KC`)
- `season` - NFL season year
- `week` - Week number
- `away_team` - Away team abbreviation
- `home_team` - Home team abbreviation
- `bk_line` - Ball Knower predicted spread (negative = home favored)
- `vegas_line` - Closing spread from data source
- `edge` - Difference (bk_line - vegas_line)

**Required data files** (must exist in `data/current_season/`):
- `power_ratings_nfelo_{season}_week_{week}.csv`
- `power_ratings_substack_{season}_week_{week}.csv`
- `epa_tiers_nfelo_{season}_week_{week}.csv`
- `qb_epa_substack_{season}_week_{week}.csv`
- `weekly_projections_ppg_substack_{season}_week_{week}.csv` (contains matchups & Vegas lines)
- `strength_of_schedule_nfelo_{season}_week_{week}.csv`

### Run Tests

```bash
python test_data_loading.py
```

This validates that all data files load correctly with proper team name normalization.

## Project Structure

```
BK_Build/
├── data/
│   ├── current_season/          # Weekly nfelo & Substack CSVs (Week 11)
│   └── reference/               # Head coaches, AV data
├── src/                         # Core Python modules
│   ├── __init__.py
│   ├── config.py               # Single source of truth for all settings
│   ├── team_mapping.py         # Normalize team names across data sources
│   ├── data_loader.py          # Load nfelo, Substack, nfl_data_py
│   ├── features.py             # Leak-free rolling EPA features
│   ├── models.py               # v1.0, v1.1, v1.2 spread models + backtest
│   └── betting_utils.py        # EV, Kelly, probability utilities
├── ball_knower/                 # Unified data loading package
│   └── io/
│       └── loaders.py          # Unified loaders for all data sources
├── notebooks/                   # Jupyter notebooks
│   └── ball_knower_demo.ipynb  # Quick start demo
├── output/                      # Model predictions and backtest results
│   └── calibrated_weights_v1.json  # Calibrated model weights
├── run_weekly_predictions.py   # CLI for weekly predictions
├── run_demo.py                 # Quick demo script
├── backtest_v1_2.py            # Professional backtest with EV analysis
├── calibrate_v1_json.py        # Generate calibrated weights JSON
├── test_data_loading.py        # Data loading validation tests
└── README.md                    # This file
```

## Data Sources

### Current Week Data (Week 11, 2025)

**nfelo files** (`data/current_season/`):
- `power_ratings_nfelo_2025_week_11.csv` - ELO ratings, QB adjustments
- `epa_tiers_nfelo_2025_week_11.csv` - EPA per play (offense/defense)
- `strength_of_schedule_nfelo_2025_week_11.csv` - SOS metrics
- `nfelo_qb_rankings_2025_week_11.csv` - QB performance rankings (reference)

**Substack files** (`data/current_season/`):
- `power_ratings_substack_2025_week_11.csv` - Offensive/Defensive/Overall ratings
- `qb_epa_substack_2025_week_11.csv` - QB-level EPA data
- `weekly_projections_ppg_substack_2025_week_11.csv` - Game projections & spreads

**Reference data** (`data/reference/`):
- `nfl_head_coaches.csv` - Coach stats and tenure
- `nfl_AV_data_through_2024.xlsx` - Approximate Value through 2024

### Historical Data

Historical game data (schedules, scores, Vegas lines) is loaded from **nfl_data_py**:
- Covers 1999-2024 seasons
- Includes spreads, totals, weather, stadium info
- Used for backtesting and feature engineering

## 🔧 How It Works

### Model Progression

**v1.0 - Deterministic Baseline**
- EPA differential × 100 → point spread contribution
- nfelo differential × 0.04 → point spread contribution
- Substack overall rating differential
- Home field advantage (2.5 points)
- **No ML, just weighted combination**

**v1.1 - Enhanced Features** (not yet implemented)
- Adds: rest advantage, recent form, QB adjustments
- Still deterministic, no training required

**v1.2 - ML Correction**
- Small Ridge regression layer on top of v1.1
- Learns residual corrections from historical games
- Captures nonlinear effects v1.1 misses

### Spread Convention

All spreads are from **HOME TEAM perspective**:
- **Negative** = home team favored (e.g., -3 means home by 3)
- **Positive** = home team underdog (e.g., +3 means away by 3)

### Edge Calculation

```
Edge = Ball Knower Prediction - Vegas Line
```

- **Negative edge**: Model likes home team more than Vegas → bet home
- **Positive edge**: Model likes away team more than Vegas → bet away
- Only bet when `|Edge| >= 0.5 points` (configurable)

## Testing & Validation

### Leakage Prevention

All features are **strictly leak-free**:
- Rolling averages use `.shift(1)` to exclude current game
- No future information in any feature
- Date-based validation ensures proper time ordering

### Data Integrity

Team name normalization handles all variations:
- nfelo: `LAR`, `KC`, `BUF`
- Substack power: `Rams`, `Chiefs`, `Bills`
- Substack QB: `ram`, `kan`, `buf` (lowercase)
- Substack weekly: `Los Angeles Rams`, `Kansas City Chiefs`

All map to nfl_data_py standard: `LAR`, `KC`, `BUF`

## 📊 Next Steps

### To Build Full Backtest System:

1. **Load historical schedules** (2015-2024)
   ```python
   from src import data_loader
   schedules = data_loader.load_historical_schedules(2015, 2024)
   ```

2. **Engineer features** (leak-free rolling EPA)
   ```python
   from src import features
   schedules_with_features = features.engineer_all_features(schedules, weekly_stats)
   ```

3. **Train v1.2 model**
   ```python
   from src import models
   model = models.MLCorrectionModel()
   model.fit(X_train, y_train)
   ```

4. **Backtest across seasons**
   ```python
   metrics = models.backtest_model(model, train_df, test_df)
   ```

5. **Analyze ROI by edge bin**
   ```python
   roi_df = models.calculate_roi_by_edge(actuals, predictions, vegas, edge_bins=[0.5, 1, 2, 3, 5])
   ```

### To Run Weekly Predictions:

1. Download latest CSVs from nfelo and Substack
2. Place files in `data/current_season/` with correct naming:
   - `power_ratings_nfelo_{season}_week_{week}.csv`
   - `power_ratings_substack_{season}_week_{week}.csv`
   - `epa_tiers_nfelo_{season}_week_{week}.csv`
   - `qb_epa_substack_{season}_week_{week}.csv`
   - `weekly_projections_ppg_substack_{season}_week_{week}.csv`
   - `strength_of_schedule_nfelo_{season}_week_{week}.csv`
3. Run the weekly predictions CLI:
   ```bash
   python run_weekly_predictions.py --season 2025 --week {week}
   ```
4. Review output CSV in `output/predictions_{season}_week_{week}.csv`
5. Compare predictions to live betting lines
6. Bet where edge exists (edge >= 0.5 points recommended)

## 🐛 Known Issues & Limitations

### Data Availability

- nfelo/Substack data only available for current week (Week 11, 2025)
- Historical ratings from these sources not available
- **Solution**: Build v1.0-v1.2 using nfl_data_py historical EPA data first, then layer in external ratings for current week predictions

### Model Status

- v1.0 baseline complete and tested
- v1.1 enhanced features - framework built, needs implementation
- v1.2 ML correction - code written, needs training on historical data
- Backtesting - needs historical feature engineering completed

### Data Quirks

- Substack files have 2-row headers (handled)
- Some matchups use "at", some use "vs" (handled)
- QBs with multiple teams like "cle, cin" (take first team)
- File naming: `nfelo_nfl_win_totals_2025_week_11 (1).csv` (duplicate upload?)

## Key Design Decisions

### Why Deterministic Baseline First?

Previous attempts with ChatGPT/Gemini failed because they jumped straight to complex ML without a solid foundation. The deterministic v1.0:
- Forces us to understand what actually predicts spreads
- Provides interpretable baseline for comparison
- Catches data issues early (garbage in = garbage out)
- Often performs surprisingly well (simplicity is underrated)

### Why Not Just Use ML?

Pure ML models tend to:
- Overfit on noise rather than signal
- Fail to generalize across seasons/rule changes
- Lack interpretability (can't explain why)
- Struggle with small data (16-17 games/season/team)

Our hybrid approach:
- Deterministic core (domain knowledge)
- Small ML layer (capture residual patterns)
- Best of both worlds

### Why Leak Prevention is Critical

A single leaked feature (e.g., using same-week EPA instead of past EPA) can:
- Make backtest look amazing
- Completely fail in live betting
- Waste months of work

We validate every feature to prevent this.

## Contributing

This is a personal betting system, but the code structure follows best practices:

- **Modular**: Each file has single responsibility
- **Tested**: Data loading validated with actual files
- **Documented**: Every function has docstrings
- **Type-safe**: Sensible parameter types
- **DRY**: Single config file, no duplicates

## Lessons from Previous Attempts

### What Failed Before (ChatGPT/Gemini):

1. Multiple conflicting config blocks
2. Team name mismatches in merges
3. Leakage in rolling stats
4. Week alignment errors
5. Variables referenced before definition
6. Over-reliance on notebook state

### What We Fixed:

1. Single `config.py` file
2. Comprehensive team name mapping
3. `.shift(1)` in all rolling calculations
4. Deterministic, testable code
5. Proper module imports
6. Fresh execution from source

## Resources

- **nfl_data_py**: https://github.com/cooperdff/nfl_data_py
- **nfelo**: https://www.nfeloapp.com
- **Substack modelers**: (various independent analysts)

## License

Personal use only. Not financial advice. Gamble responsibly.

---

**Status**: Core system built, ready for historical backtesting and weekly prediction workflow.

**Author**: Built with Claude (Anthropic)

**Last Updated**: November 16, 2025
