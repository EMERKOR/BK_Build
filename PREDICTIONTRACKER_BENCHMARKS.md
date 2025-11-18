# PredictionTracker Benchmarks for Ball Knower

This document describes the new PredictionTracker benchmark layer added to Ball Knower.

## Overview

The PredictionTracker benchmark module allows you to:
1. Compare Ball Knower's spreads against the "crowd of models" from PredictionTracker
2. Compute accuracy metrics (MAE) for BK, PT consensus, and Vegas
3. Flag outlier games where BK disagrees significantly with the consensus
4. Export detailed analysis to CSV for further investigation

**Important**: This is strictly for benchmarking and sanity checking. PredictionTracker data is NOT used as model features for training.

## New Files Created

### Core Modules

1. **`ball_knower/datasets/v1_2.py`**
   - Provides `build_training_frame()` to load canonical BK v1.2 game-level data
   - Loads nfelo historical games with features and outcomes
   - Returns DataFrame with: season, week, teams, scores, margins, Vegas lines, ELO, etc.

2. **`ball_knower/benchmarks/predictiontracker.py`**
   - `load_predictiontracker_csv()`: Loads and normalizes PredictionTracker CSV files
   - `merge_with_bk_games()`: Merges PT predictions with BK canonical games
   - `compute_summary_metrics()`: Computes MAE metrics and outlier statistics
   - Reuses existing team normalization from `src/team_mapping.py`

### CLI Script

3. **`src/run_predictiontracker_benchmarks.py`**
   - Executable CLI script to run benchmarks
   - Loads BK v1.2 data, merges with PT predictions
   - Computes metrics and flags outliers
   - Writes CSV outputs and prints text summary

### Documentation & Structure

4. **`data/benchmarks/`** - Output directory for benchmark results
5. **`data/external/`** - Directory for manually downloaded PT CSV files
6. **README files** in both directories with usage instructions

## Installation & Setup

### Prerequisites

Ensure pandas and numpy are installed (already required by existing BK scripts):

```bash
pip install pandas numpy
```

### Download PredictionTracker Data

1. Manually download PredictionTracker NFL prediction CSV files
2. Place them in `data/external/`:
   ```
   data/external/predictiontracker_nfl_2024.csv
   data/external/predictiontracker_nfl_2023.csv
   ```

**Expected CSV Format:**
- Home/Away team columns (auto-detected: "home", "visitor", "away", etc.)
- Vegas line (optional: "line", "vegasline", "spread")
- Prediction average or individual model predictions
- Prediction standard deviation (optional: "predictionstd", "std")

The loader will auto-detect column names and normalize team names using BK's canonical mapping.

## Usage

### Basic Benchmark

```bash
python src/run_predictiontracker_benchmarks.py \
    --pt_csv data/external/predictiontracker_nfl_2024.csv
```

### Custom Options

```bash
python src/run_predictiontracker_benchmarks.py \
    --pt_csv data/external/predictiontracker_nfl_2024.csv \
    --output_dir output/benchmarks \
    --outlier_threshold 5.0
```

**Parameters:**
- `--pt_csv`: Path to PredictionTracker CSV (required)
- `--output_dir`: Output directory (default: `data/benchmarks`)
- `--outlier_threshold`: Outlier threshold in points (default: 4.0)

### Programmatic Usage

```python
from ball_knower.benchmarks import predictiontracker as pt_bench
from ball_knower.datasets import v1_2

# Load BK canonical games
games = v1_2.build_training_frame()

# Load PredictionTracker predictions
pt_df = pt_bench.load_predictiontracker_csv("data/external/predictiontracker_nfl_2024.csv")

# Merge and compute metrics
merged = pt_bench.merge_with_bk_games(
    pt_df,
    bk_games=games,
    outlier_threshold=4.0
)

# Compute summary
summary = pt_bench.compute_summary_metrics(merged)
print(summary)

# Analyze outliers
outliers = merged[merged['bk_outlier_flag'] == True]
print(f"Found {len(outliers)} outlier games")
```

## Output Files

Running the benchmark creates two CSV files in `data/benchmarks/`:

### 1. `predictiontracker_merged_{season}.csv`

Full merged dataset with all games and metrics:

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier |
| `season`, `week` | Game season and week |
| `home_team`, `away_team` | Normalized team codes |
| `home_score`, `away_score` | Final scores |
| `home_margin` | Home score - away score |
| `vegas_line` | Vegas closing line (home referenced) |
| `pt_pred_avg` | PredictionTracker consensus spread |
| `pt_pred_std` | Standard deviation of model predictions |
| `pt_mae_vs_margin` | \|home_margin - pt_pred_avg\| |
| `bk_mae_vs_margin` | \|home_margin - bk_line\| (if BK predictions available) |
| `vegas_mae_vs_margin` | \|home_margin - vegas_line\| |
| `bk_vs_pt_diff` | bk_line - pt_pred_avg |
| `bk_outlier_flag` | TRUE if \|bk_vs_pt_diff\| > threshold |

### 2. `predictiontracker_summary_{season}.csv`

Summary metrics:

| Metric | Description |
|--------|-------------|
| `n_games` | Total number of games |
| `n_games_with_pt` | Games with PT predictions |
| `pt_mae_vs_margin` | Mean absolute error: PT vs actual margin |
| `bk_mae_vs_margin` | MAE: BK vs actual margin |
| `vegas_mae_vs_margin` | MAE: Vegas vs actual margin |
| `bk_vs_pt_mean_diff` | Mean difference: BK - PT |
| `bk_vs_pt_mae_diff` | Mean absolute difference: BK vs PT |
| `bk_outlier_count` | Number of outlier games |
| `bk_outlier_pct` | Percentage of games flagged as outliers |

## Example Output

```
================================================================================
BALL KNOWER vs PREDICTIONTRACKER BENCHMARK
================================================================================

[1/4] Loading canonical Ball Knower v1.2 game-level frame...
  ✓ Loaded 4,345 games from 2009-2025

[2/4] Loading PredictionTracker predictions from: data/external/pt_2024.csv
  ✓ Loaded 267 PredictionTracker predictions
  ✓ Model disagreement (std) available

[3/4] Merging frames and computing benchmark metrics...
  ✓ Merged 4,345 games
  ✓ 267 games (6.1%) have PT predictions

[4/4] Computing summary metrics...
  ✓ Wrote merged data: data/benchmarks/predictiontracker_merged_2009-2025.csv
  ✓ Wrote summary: data/benchmarks/predictiontracker_summary_2009-2025.csv

================================================================================
BENCHMARK SUMMARY
================================================================================

Games analyzed: 4,345
Games with PT predictions: 267

--- Mean Absolute Error vs Actual Margin ---
  PredictionTracker: 10.45 points
  Ball Knower v1.2:  10.32 points
  Vegas closing:     10.21 points

--- BK vs PT Comparison ---
  Mean difference (BK - PT): +0.12 points
  Mean absolute difference:  2.34 points

--- BK Outlier Analysis (threshold: 4.0 pts) ---
  Outlier games: 23 (8.6%)
```

## Use Cases

### 1. Sanity Checking
Identify games where BK disagrees significantly with the model consensus:

```python
outliers = merged[merged['bk_outlier_flag'] == True].sort_values('bk_vs_pt_diff', ascending=False)
print(outliers[['game_id', 'home_team', 'away_team', 'bk_line', 'pt_pred_avg', 'bk_vs_pt_diff']])
```

### 2. Model Validation
Compare BK's accuracy against PredictionTracker consensus:

```python
print(f"BK MAE: {merged['bk_mae_vs_margin'].mean():.2f}")
print(f"PT MAE: {merged['pt_mae_vs_margin'].mean():.2f}")
print(f"Vegas MAE: {merged['vegas_mae_vs_margin'].mean():.2f}")
```

### 3. Edge Discovery
Find games where BK and PT diverge (potential value):

```python
large_diff = merged[merged['bk_vs_pt_diff'].abs() > 3.0]
print(f"Found {len(large_diff)} games with BK-PT difference > 3 points")
```

### 4. Bias Analysis
Understand systematic differences between BK and PT:

```python
print(f"Mean BK - PT difference: {merged['bk_vs_pt_diff'].mean():.2f}")
# Positive = BK more bullish on home teams
# Negative = BK more bearish on home teams
```

## Team Normalization

The module reuses BK's existing canonical team mapping from `src/team_mapping.py`. Supported variations include:

- Standard abbreviations: KC, BUF, LAR, etc.
- Full names: Kansas City Chiefs, Buffalo Bills
- Nicknames: Chiefs, Bills, Rams
- Historical: OAK → LV, STL → LAR, SD → LAC
- Case-insensitive: kan, buf, lar, etc.

If a team name in the PT CSV cannot be mapped, the loader will raise a clear error with the unknown team names.

## Technical Notes

### Column Detection

The PT CSV loader uses fuzzy column detection to handle various CSV formats:

| Expected Column | Detected Names |
|----------------|----------------|
| Home team | "home", "hometeam", "home_team" |
| Away team | "visitor", "away", "awayteam", "road" |
| Vegas line | "line", "vegasline", "spread", "closingline" |
| PT average | "predictionavg", "predictavg", "predavg", "consensus" |
| PT std | "predictionstd", "predstd", "std" |

If no explicit average column exists, the loader computes consensus from all numeric model columns.

### Merge Strategy

Games are merged on `(home_team, away_team)`. Future enhancements could add date-based matching for tighter joins.

### Missing Data

- Games without PT predictions will have NaN in PT columns
- Games without BK predictions will skip BK metrics
- All handling is graceful with clear warnings

## Future Enhancements

Potential improvements (not implemented):

1. **Date-based merging**: Use game dates for tighter matching
2. **Against-the-spread (ATS) metrics**: Track cover rates vs closing lines
3. **Time-series analysis**: Track BK vs PT divergence over time
4. **Confidence intervals**: Use PT std to compute z-scores for BK outliers
5. **Web scraping**: Auto-download PT data (requires additional dependencies)

## Dependencies

No new dependencies beyond what BK already requires:
- pandas (already used throughout BK)
- numpy (already used throughout BK)

All code is self-contained and uses existing BK utilities (team mapping, etc.)

## File Structure Summary

```
ball_knower/
├── benchmarks/
│   ├── __init__.py
│   └── predictiontracker.py     # Core benchmark logic
├── datasets/
│   ├── __init__.py
│   └── v1_2.py                  # Canonical dataset builder
└── io/
    └── loaders.py               # Existing data loaders

src/
├── team_mapping.py              # Existing team normalization (reused)
└── run_predictiontracker_benchmarks.py  # CLI script

data/
├── benchmarks/                  # Benchmark output directory
│   └── README.md
└── external/                    # Manual download directory
    └── README.md
```

## Questions?

For issues or enhancements, update the PredictionTracker module or consult the inline docstrings in:
- `ball_knower/benchmarks/predictiontracker.py`
- `ball_knower/datasets/v1_2.py`
