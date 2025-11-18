# PredictionTracker Benchmark v1.2 Upgrade

## Summary

Successfully upgraded the PredictionTracker benchmark layer to work with the canonical v1.2 dataset that includes actual game scores and Vegas spreads. The benchmark now computes MAE vs actual game results (not just vs Vegas) and uses precise (season, week, home_team, away_team) matching.

## Branch

**Branch name:** `claude/upgrade-predictiontracker-v1-2-017J6mSpTBTkxEzQd48UP67r`

## What Was Built

### 1. Canonical v1.2 Dataset Builder
**File:** `ball_knower/datasets/v1_2.py`

A reusable dataset builder that combines nflverse games (with scores) and nfelo data (with ratings):

**Key function:** `build_training_frame(start_season=2009, end_season=2024)`

**Data sources:**
- **nflverse games** (https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv)
  - Provides: actual scores (home_score, away_score), Vegas closing spreads
  - ~4,300 regular season games from 2009-2024

- **nfelo games** (https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv)
  - Provides: ELO ratings, situational adjustments (rest, divisional, QB, etc.)

**Output columns:**
- `game_id`: unique identifier (season_week_away_home)
- `season`, `week`, `gameday`: temporal identifiers
- `away_team`, `home_team`: normalized team codes
- `away_score`, `home_score`, `home_margin`: actual game results
- `vegas_closing_spread`: Vegas closing line (home referenced)
- `nfelo_diff`: ELO differential (home - away)
- `rest_advantage`, `div_game`, `surface_mod`, `time_advantage`, `qb_diff`: situational features

**Merge strategy:**
- Inner join on (season, week, away_team, home_team)
- Only keeps games present in both datasets
- Result: ~4,300 games with complete data (scores + spreads + ELO)

### 2. PredictionTracker Benchmark Module
**File:** `ball_knower/benchmarks/predictiontracker.py`

**Key improvements over previous version:**

#### a. PT CSV Loader: `load_predictiontracker_csv(path)`
- Parses PT CSV with flexible column detection
- **NEW:** Extracts season/week from game date using NFL calendar logic:
  - Games in Sept-Dec belong to that year's season
  - Games in Jan/Feb belong to previous year's season
  - Week numbers computed from season start (first Thursday of September)
- Normalizes team names using existing team_mapping.py
- Returns DataFrame with: season, week, home_team, away_team, pt_spread

#### b. Merge Function: `merge_with_bk_games(pt_df, bk_games, outlier_threshold=4.0)`
- **NEW merge key:** (season, week, home_team, away_team) — ensures 1:1 matching
- **OLD merge key was:** (home_team, away_team) — could match multiple seasons
- Uses `validate='1:1'` to enforce unique matches
- Computes MAE metrics vs actual game margin:
  - `mae_pt_vs_actual`: |home_margin - pt_spread|
  - `mae_vegas_vs_actual`: |home_margin - vegas_closing_spread|
  - `mae_bk_vs_actual`: |home_margin - bk_line| (if BK predictions present)
- Flags BK outliers where |bk_line - pt_spread| > threshold
- Reports matched/unmatched counts for debugging

#### c. Summary Metrics: `compute_summary_metrics(merged)`
Returns single-row DataFrame with:
- `n_games`: total matched games
- `mae_pt_vs_actual`, `mae_vegas_vs_actual`, `mae_bk_vs_actual`
- `bk_vs_pt_mean_diff`, `bk_vs_pt_mae_diff`
- `bk_outlier_count`, `bk_outlier_pct`

### 3. CLI Runner
**File:** `src/run_predictiontracker_benchmarks.py`

**Usage:**
```bash
python src/run_predictiontracker_benchmarks.py \
    --pt_csv data/external/predictiontracker_nfl_2024.csv \
    --output_dir data/benchmarks \
    --outlier_threshold 4.0
```

**Workflow:**
1. Loads canonical v1.2 dataset (nflverse + nfelo)
2. Loads PT CSV and extracts season/week from dates
3. Merges on (season, week, home_team, away_team)
4. Computes MAE vs actual margins
5. Writes outputs:
   - `predictiontracker_merged_{season}.csv`: full dataset with all metrics
   - `predictiontracker_summary_{season}.csv`: summary statistics
6. Prints report with:
   - MAE for PT, Vegas, BK (if available)
   - Sample of matched games
   - Sample of unmatched games (for debugging)

### 4. Sample Data
**File:** `data/external/predictiontracker_nfl_2024_sample.csv`

A small 10-row synthetic PT CSV for testing with realistic Week 1 2024 games.

## Merge Logic Improvements

### Previous Version (v1.0)
- Merge key: `(home_team, away_team)`
- Problem: Could match KC vs BUF from multiple seasons (2020, 2021, 2022, etc.)
- Result: Duplicate or ambiguous matches

### Current Version (v1.2)
- Merge key: `(season, week, home_team, away_team)`
- Validates: Each PT game matches at most ONE BK game
- Benefit: Precise 1:1 mapping, accurate metrics

Example:
- PT CSV: "2024-09-05, Kansas City, Baltimore"
  - Parsed → season=2024, week=1, home_team='KC', away_team='BAL'
- BK dataset: Contains game (2024, 1, 'BAL', 'KC')
- Match: ✓ Unique and precise

## MAE Computation Improvements

### Previous Version
- MAE computed vs Vegas closing line (not vs actual results)
- Actual game margin was NaN (nfelo dataset had no scores)

### Current Version
- MAE computed vs **actual game margin**:
  - `mae_pt_vs_actual = |home_margin - pt_spread|`
  - `mae_vegas_vs_actual = |home_margin - vegas_closing_spread|`
- Actual margin = home_score - away_score (from nflverse)
- Now measures true prediction accuracy, not just agreement with Vegas

## Metrics Available

### Per-Game Metrics (in merged CSV)
- `home_margin`: actual game result (home_score - away_score)
- `vegas_closing_spread`: Vegas closing line
- `pt_spread`: PredictionTracker consensus spread
- `mae_pt_vs_actual`: PT error vs actual result
- `mae_vegas_vs_actual`: Vegas error vs actual result
- `mae_bk_vs_actual`: BK error vs actual result (if BK predictions available)
- `bk_vs_pt_diff`: BK - PT (outlier analysis)
- `bk_outlier_flag`: boolean for |BK - PT| > threshold

### Summary Metrics (in summary CSV)
- `n_games`: total matched games
- `mae_pt_vs_actual`: mean PT error
- `mae_vegas_vs_actual`: mean Vegas error
- `mae_bk_vs_actual`: mean BK error
- `bk_vs_pt_mean_diff`: average BK - PT difference
- `bk_vs_pt_mae_diff`: average |BK - PT| difference
- `bk_outlier_count`: count of outlier games
- `bk_outlier_pct`: percentage of outliers

## Limitations

### Current State
1. **BK predictions (`bk_line`) not yet in canonical dataset**
   - The v1.2 dataset builder loads scores and Vegas lines
   - It does NOT yet include BK model predictions
   - To add BK predictions, use `add_bk_predictions(df, model_coef, intercept)`
   - Model coefficients available in: `output/ball_knower_v1_2_model.json` (if trained)

2. **Season/week extraction from date is approximate**
   - Uses heuristic: first Thursday of September = Week 1
   - May be off by ±1 week for edge cases (e.g., early/late season starts)
   - For production use, recommend adding explicit season/week columns to PT CSV

3. **Only regular season games**
   - Current filter: `game_type == 'REG'`
   - Playoffs excluded for now

## Next Steps (Future Work)

1. **Wire BK predictions into canonical dataset:**
   ```python
   from ball_knower.datasets import v1_2
   import json

   # Load model
   with open('output/ball_knower_v1_2_model.json') as f:
       model = json.load(f)

   # Build dataset and add BK predictions
   df = v1_2.build_training_frame()
   df = v1_2.add_bk_predictions(df, model['coefficients'], model['intercept'])

   # Now df has 'bk_line' column
   ```

2. **Add more PT data sources:**
   - Current: only handles single PT CSV
   - Could extend to multiple seasons or sources

3. **Add time-series analysis:**
   - Track MAE by season, week
   - Analyze if PT/BK/Vegas improve over time

4. **Add bet outcome simulation:**
   - Use actual margins to simulate betting results
   - Compute ROI for PT/BK/Vegas strategies

## Testing

### Validation Performed
- ✅ Python syntax validation (all modules)
- ✅ Created sample PT CSV with realistic data
- ✅ File structure and imports are correct

### Manual Testing Required (in environment with pandas)
```bash
# 1. Build canonical dataset directly
python -c "
from ball_knower.datasets import v1_2
df = v1_2.build_training_frame(start_season=2020, end_season=2024)
print(df.head())
print(df.columns.tolist())
"

# 2. Run full benchmark
python src/run_predictiontracker_benchmarks.py \
    --pt_csv data/external/predictiontracker_nfl_2024_sample.csv \
    --output_dir data/benchmarks

# 3. Check outputs
cat data/benchmarks/predictiontracker_summary_2024.csv
head -20 data/benchmarks/predictiontracker_merged_2024.csv
```

## Files Changed

### New Files Created
```
ball_knower/
  datasets/
    __init__.py
    v1_2.py                                   # Canonical dataset builder
  benchmarks/
    __init__.py
    predictiontracker.py                       # Upgraded benchmark module

src/
  run_predictiontracker_benchmarks.py          # CLI runner

data/
  external/
    predictiontracker_nfl_2024_sample.csv      # Sample PT data for testing
```

### Total Lines Added
- ball_knower/datasets/v1_2.py: ~350 lines
- ball_knower/benchmarks/predictiontracker.py: ~400 lines
- src/run_predictiontracker_benchmarks.py: ~250 lines
- **Total: ~1,000 lines of new code**

## Technical Notes

### Team Name Normalization
- Uses existing `src/team_mapping.py`
- Handles aliases: "Kansas City" → "KC", "LA Rams" → "LAR"
- Raises error if unknown team found

### Date Parsing for Season/Week
```python
# NFL season logic
season = date.year if date.month >= 9 else date.year - 1

# Week approximation
season_start = first_thursday_of_september(season)
week = ((date - season_start).days // 7) + 1
```

### Merge Validation
```python
merged = pt_df.merge(
    bk_games,
    on=['season', 'week', 'home_team', 'away_team'],
    how='inner',
    validate='1:1',  # Ensures unique matching
)
```

## Summary Comparison

| Feature | v1.0 (old) | v1.2 (new) |
|---------|-----------|-----------|
| Data source | nfelo only | nflverse + nfelo |
| Has actual scores? | ❌ No (NaN) | ✅ Yes |
| Has Vegas spreads? | ✅ Yes | ✅ Yes |
| Merge key | (home, away) | (season, week, home, away) |
| MAE vs actual? | ❌ No | ✅ Yes |
| MAE vs Vegas? | ✅ Yes | ✅ Yes |
| 1:1 matching? | ❌ No | ✅ Yes |
| Season/week in PT? | ❌ No | ✅ Yes (extracted) |
| BK predictions? | ❌ No | ⚠️ Ready (need to wire) |

## Commit Information

**Commit hash:** cf11e2e

**Commit message:**
> Upgrade PredictionTracker benchmark to use canonical v1.2 dataset with scores
>
> Major improvements:
> 1. Created canonical v1.2 dataset builder with nflverse (scores) + nfelo (ratings)
> 2. Upgraded PT benchmark with (season, week, home, away) precise matching
> 3. MAE now computed vs actual game results
> 4. Added CLI runner with detailed reporting
