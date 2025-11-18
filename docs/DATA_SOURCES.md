# Ball Knower Data Sources

This document describes all data sources used by Ball Knower, file naming conventions, and which modules consume which datasets.

## File Naming Conventions

### Current Season Data (Category-First)

The **canonical naming convention** for current-season CSV files is:

```
{category}_{provider}_{season}_week_{week}.csv
```

**Examples**:
- `power_ratings_nfelo_2025_week_11.csv`
- `qb_epa_substack_2025_week_11.csv`
- `weekly_projections_ppg_substack_2025_week_11.csv`

**Supported Categories**:
- `power_ratings` - Overall team power/strength ratings
- `epa_tiers` - Expected Points Added offensive/defensive metrics
- `strength_of_schedule` - SOS rankings and metrics
- `qb_epa` - Quarterback-level EPA and performance stats
- `weekly_projections_ppg` - Game projections and point spreads

**Supported Providers**:
- `nfelo` - [nfeloapp.com](https://www.nfeloapp.com) ratings and analytics
- `substack` - Various Substack-based NFL analysts and modelers

### Legacy Naming (Provider-First)

For backward compatibility, the loader also supports:

```
{provider}_{category}_{season}_week_{week}.csv
```

**Examples**:
- `nfelo_qb_rankings_2025_week_11.csv`
- `substack_weekly_proj_elo_2025_week_11.csv`

The unified loader (`ball_knower.io.loaders`) will automatically try category-first, then fall back to provider-first if needed.

## Primary Data Sources

### 1. nflverse (Historical Games & Schedules)

**Source**: [nfl_data_py](https://github.com/cooperdff/nfl_data_py) Python package

**Coverage**: 1999-2025 NFL seasons

**Key Datasets**:
- `schedules.parquet` - All games with dates, teams, scores, lines
- `games.csv` - Game-level results and metadata
- Historical team stats (offense, defense, special teams)
- Play-by-play data (for EPA calculations)

**Fields Used**:
- `game_id` - Unique game identifier
- `season`, `week` - Temporal context
- `home_team`, `away_team` - Team abbreviations (standard format)
- `home_score`, `away_score` - Final scores
- `spread_line` - Vegas closing spread
- `total_line` - Over/under total
- `stadium`, `roof`, `surface` - Venue details
- `temp`, `wind` - Weather conditions

**Consumed By**:
- `backtest_v1_0.py` - Historical game results for backtesting
- `backtest_v1_2.py` - Training data for spread prediction
- `src/nflverse_data.py` - Data loader wrapper

### 2. nfelo (Power Ratings & Lines)

**Source**: [nfeloapp.com](https://www.nfeloapp.com) - Greer Rutten's NFL Elo system

**Coverage**: 2009-2025 (4,500+ games with historical Elo ratings)

**Current-Week Files**:
- `power_ratings_nfelo_2025_week_11.csv`
  - Columns: `team`, `elo_rating`, `qb_adj`, `overall_rating`
  - Updated weekly with latest Elo calculations

- `epa_tiers_nfelo_2025_week_11.csv`
  - Columns: `team`, `off_epa_per_play`, `def_epa_per_play`
  - EPA tiers (S+, A, B, C, D)

- `strength_of_schedule_nfelo_2025_week_11.csv`
  - Columns: `team`, `sos_rating`, `remaining_sos`
  - Strength of schedule metrics

- `nfelo_qb_rankings_2025_week_11.csv` (reference only)
  - QB performance rankings and stats

**Historical Dataset**:
- `nfelo_games.csv` (via GitHub)
  - URL: `https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv`
  - Contains historical Elo ratings and Vegas lines for backtesting

**Fields Used**:
- `elo_rating_pre` - Team Elo before the game
- `qb_adj_elo` - QB-adjusted Elo
- `home_line_close` - Vegas closing spread
- Offensive/defensive EPA metrics

**Consumed By**:
- `ball_knower.io.loaders.load_power_ratings()` - Current week predictions
- `backtest_v1_0.py` - Historical Elo-based spread modeling
- `backtest_v1_2.py` - Feature engineering for ML model

### 3. Substack Power Ratings & QB Metrics

**Source**: Various independent NFL analysts publishing on Substack

**Current-Week Files**:
- `power_ratings_substack_2025_week_11.csv`
  - Columns: `team`, `off_rating`, `def_rating`, `overall_rating`
  - Composite power rankings

- `qb_epa_substack_2025_week_11.csv`
  - Columns: `team_abbr`, `qb_name`, `epa_per_play`, `completions`, `attempts`
  - QB-level EPA and passing stats

- `weekly_projections_ppg_substack_2025_week_11.csv`
  - Columns: `matchup`, `home_team`, `away_team`, `projected_spread`, `projected_total`
  - Game-level projections and spreads

**Team Name Quirks**:
- Power ratings use full names: `"Kansas City Chiefs"`, `"Los Angeles Rams"`
- QB data uses lowercase abbrevs: `"kan"`, `"ram"`, `"buf"`
- All normalized to standard `nfl_data_py` format: `"KC"`, `"LAR"`, `"BUF"`

**Consumed By**:
- `ball_knower.io.loaders.load_power_ratings()` - Merges with nfelo ratings
- `ball_knower.io.loaders.load_qb_metrics()` - QB-level features
- `predict_current_week.py` - Live weekly predictions

### 4. Reference Data

**Location**: `data/reference/`

**Files**:
- `nfl_head_coaches.csv`
  - Columns: `team`, `coach_name`, `tenure_years`, `win_pct`
  - Historical coach performance (not currently used in models)

- `nfl_AV_data_through_2024.xlsx`
  - Approximate Value (AV) player stats through 2024
  - Potential future feature for roster strength

**Status**: Available but not integrated into v1.0-v1.2 models yet.

## Module-to-Source Mapping

### v1.0 (Actual Margin Prediction)

**Data Sources**:
- nflverse historical games (scores, dates)
- nfelo historical ratings and EPA
- Structural features (home/away, rest, division)

**Loader Module**: `src/data_loader.py` + `backtest_v1_0.py`

**Key Merges**:
```python
# Load historical games
schedules = nfl_data_py.import_schedules(years=range(2009, 2025))

# Load nfelo ratings
nfelo_df = pd.read_csv('https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv')

# Merge on game_id
merged = schedules.merge(nfelo_df, on='game_id', how='inner')

# Calculate target: actual_margin = home_score - away_score
merged['actual_margin'] = merged['home_score'] - merged['away_score']
```

### v1.2 (Vegas Spread Prediction)

**Data Sources**:
- All v1.0 sources (nflverse + nfelo)
- QB metrics (Substack EPA, QBR)
- Structural flags (season, week, neutral site)
- Vegas closing lines (from nfelo historical data)

**Loader Module**: `src/data_loader.py` + `backtest_v1_2.py`

**Key Merges**:
```python
# Start with v1.0 base
base_df = build_v1_0_features()

# Add QB metrics
qb_df = load_qb_metrics(season, week)
merged = base_df.merge(qb_df, on=['team', 'season', 'week'], how='left')

# Add structural features
merged['rest_days'] = merged.groupby('team')['gameday'].diff().dt.days.fillna(7)
merged['is_division_game'] = (merged['home_division'] == merged['away_division']).astype(int)

# Target: vegas_spread_close (already in nfelo data)
```

### Current Week Predictions

**Data Sources**:
- Current-week CSVs from `data/current_season/`
- nfelo power ratings, EPA tiers, SOS
- Substack power ratings, QB EPA, projections

**Loader Module**: `ball_knower.io.loaders`

**Example**:
```python
from ball_knower.io import loaders

# Load all sources for Week 11, 2025
data = loaders.load_all_sources(season=2025, week=11)

# Access individual datasets
power_ratings = data['power_ratings']  # Combined nfelo + Substack
qb_metrics = data['qb_metrics']        # Substack QB EPA
projections = data['projections']      # Substack weekly projections
```

## Data Quality and Normalization

### Team Name Normalization

All team names are normalized to `nfl_data_py` standard abbreviations:

| Source Format | nfl_data_py Standard |
|---------------|---------------------|
| `Kansas City Chiefs` | `KC` |
| `Los Angeles Rams` | `LAR` |
| `kan` (lowercase) | `KC` |
| `ram` (lowercase) | `LAR` |

**Handled By**: `src/team_mapping.py` - `normalize_team_name()`

### Missing Data Handling

- **Missing QB stats**: Filled with league-average EPA
- **Missing power ratings**: Use nfelo as primary, Substack as fallback
- **Missing lines**: Games excluded from backtest (can't calculate edge without Vegas line)

### Data Freshness

- **Historical data**: Cached locally, refreshed monthly
- **Current week**: Updated manually from nfelo/Substack before game day
- **nflverse**: Updated via `nfl_data_py` package (pulls latest from GitHub)

---

**Last Updated**: 2025-11-18
**Maintained By**: Ball Knower Development Team
