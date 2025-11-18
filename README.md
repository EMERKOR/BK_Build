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
│   └── models.py               # v1.0, v1.1, v1.2 spread models + backtest
├── notebooks/                   # Jupyter notebooks
│   └── ball_knower_demo.ipynb  # Quick start demo
├── output/                      # Model predictions and backtest results
├── test_data_loading.py        # Data loading validation tests
└── README.md                    # This file
```

## Architecture & Documentation

For detailed system documentation, see:
- **[DATA_SETUP_GUIDE.md](DATA_SETUP_GUIDE.md)** - How to add new data sources and update weekly files
- **[STEP_2_ASSESSMENT.md](STEP_2_ASSESSMENT.md)** - Feature engineering and model architecture details

### Data File Naming Convention

All current-season files in `data/current_season/` use **category-first naming**:

```
{category}_{provider}_{season}_week_{week}.csv
```

**Examples:**
- `power_ratings_nfelo_2025_week_11.csv`
- `qb_rankings_nfelo_2025_week_11.csv`
- `weekly_projections_elo_substack_2025_week_11.csv`

**Supported providers:** nfelo, substack, 538, espn, pff, gsis, user, manual

**Note:** Provider-first filenames (e.g., `nfelo_power_ratings_...`) still exist temporarily for backward compatibility but will be phased out. The unified loader in `ball_knower/io/loaders.py` automatically handles both naming conventions.

## Data Sources

### Current Week Data (Week 11, 2025)

**nfelo files** (`data/current_season/`):
- `power_ratings_nfelo_2025_week_11.csv` - ELO ratings, QB adjustments
- `epa_tiers_nfelo_2025_week_11.csv` - EPA per play (offense/defense)
- `strength_of_schedule_nfelo_2025_week_11.csv` - SOS metrics
- `qb_rankings_nfelo_2025_week_11.csv` - QB performance rankings (reference)
- `nfl_receiving_leaders_nfelo_2025_week_11.csv` - Receiving stats (reference)
- `nfl_win_totals_nfelo_2025_week_11.csv` - Season win projections (reference)

**Substack files** (`data/current_season/`):
- `power_ratings_substack_2025_week_11.csv` - Offensive/Defensive/Overall ratings
- `qb_epa_substack_2025_week_11.csv` - QB-level EPA data
- `weekly_projections_ppg_substack_2025_week_11.csv` - Game projections & spreads
- `weekly_projections_elo_substack_2025_week_11.csv` - Elo-based projections (reference)

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
2. Update `CURRENT_WEEK` in `src/config.py`
3. Replace files in `data/current_season/`
4. Run `notebooks/ball_knower_demo.ipynb`
5. Compare predictions to Bovada lines
6. Bet where edge exists!

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

- Substack files have 2-row headers (handled by loaders)
- Some matchups use "at", some use "vs" (handled)
- QBs with multiple teams like "cle, cin" (take first team)
- Legacy provider-first filenames still exist for backward compatibility

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
