# Ball Knower v1.1 - Backtesting Guide

This guide explains how to prepare weekly data and run backtests using the Ball Knower v1.1 calibrated spread model.

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Running Backtests](#running-backtests)
4. [Interpreting Results](#interpreting-results)
5. [Example Workflows](#example-workflows)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The Ball Knower v1.1 backtesting workflow consists of two main steps:

1. **Data Preparation:** Convert raw nfelo/Substack downloads into standardized file names
2. **Backtest Execution:** Calibrate weights on historical weeks and test on holdout weeks

### What You Need

- **Raw data files** from nfelo and Substack (power ratings, EPA tiers, projections)
- **Schedule data** with Vegas lines (already included in `data/cache/schedules_2025.csv`)
- **Python 3.8+** with pandas and numpy

---

## Data Preparation

### Step 1: Download Raw Files

Download the following CSVs from your data sources:

**From nfelo:**
- Power ratings (contains `team`, `nfelo` columns)
- EPA tiers - offense/defense (contains `team`, `EPA/Play`, `EPA/Play Against`)
- Strength of schedule (contains `team`, `sos` or similar)

**From Substack:**
- Power ratings (contains `team`, `Ovr.` columns)
- QB EPA rankings (contains `team`, quarterback EPA metrics)
- Weekly projections (contains matchups and spreads)

Save these files in a directory like `raw_downloads/`.

### Step 2: Prepare Weekly Ratings

Use the `prepare_weekly_ratings.py` script to convert raw files into canonical names:

```bash
python scripts/prepare_weekly_ratings.py \
    --season 2025 \
    --week 11 \
    --source-dir raw_downloads/ \
    --nfelo-power nfelo_power_ratings_2025_week_11.csv \
    --nfelo-epa nfelo_epa_tiers_off_def_2025_week_11.csv \
    --nfelo-sos nfelo_strength_of_schedule_2025_week_11.csv \
    --substack-power substack_ratings_week11.csv \
    --substack-qb substack_qb_epa_week11.csv \
    --substack-proj substack_weekly_projections_week11.csv
```

**What this does:**
- Validates that each CSV contains expected columns
- Copies files to `data/current_season/` with standardized names:
  - `power_ratings_nfelo_2025_week_11.csv`
  - `epa_tiers_nfelo_2025_week_11.csv`
  - `strength_of_schedule_nfelo_2025_week_11.csv`
  - `power_ratings_substack_2025_week_11.csv`
  - `qb_epa_substack_2025_week_11.csv`
  - `weekly_projections_ppg_substack_2025_week_11.csv`

**Repeat for each week** you want to include in your backtest (e.g., weeks 1-10 for training, 11-18 for testing).

### Expected File Locations

After preparation, your directory structure should look like:

```
BK_Build/
├── data/
│   ├── cache/
│   │   └── schedules_2025.csv              # ✓ Already present
│   └── current_season/
│       ├── power_ratings_nfelo_2025_week_1.csv
│       ├── epa_tiers_nfelo_2025_week_1.csv
│       ├── power_ratings_substack_2025_week_1.csv
│       ├── weekly_projections_ppg_substack_2025_week_1.csv
│       ├── power_ratings_nfelo_2025_week_2.csv
│       ├── ...
│       └── weekly_projections_ppg_substack_2025_week_18.csv
```

---

## Running Backtests

Once data is prepared, use the `run_backtest_v1_1.py` script to run a full backtest.

### Basic Usage

```bash
python scripts/run_backtest_v1_1.py \
    --season 2025 \
    --train-weeks 1-10 \
    --test-weeks 11-18 \
    --edge-thresholds 1,2,3,4
```

### Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--season` | Yes | NFL season year | `2025` |
| `--train-weeks` | Yes | Weeks to calibrate on | `1-10` or `1,3,5-8` |
| `--test-weeks` | Yes | Weeks to test predictions | `11-18` or `11,13,15` |
| `--edge-thresholds` | No | ATS edge thresholds (points) | `1,2,3,4` (default) |
| `--data-dir` | No | Data directory | `data/current_season/` (default) |
| `--output-dir` | No | Output directory | `output/` (default) |

### What the Script Does

1. **Calibrate weights** on training weeks using OLS regression
2. **Generate predictions** for test weeks with both v1.1 (calibrated) and v1.0 (fixed) weights
3. **Calculate metrics:**
   - MAE (Mean Absolute Error) vs Vegas
   - RMSE (Root Mean Squared Error) vs Vegas
   - Correlation with Vegas lines
4. **Analyze ATS performance** at each edge threshold:
   - Win-Loss-Push record
   - Win rate (excluding pushes)
   - ROI (Return on Investment, assuming -110 odds)
5. **Save results:**
   - `backtest_v1_1_{season}_train_{range}_test_{range}.csv` - Full game-level predictions
   - `backtest_v1_1_{season}_ats_summary.csv` - ATS performance summary

---

## Interpreting Results

### Console Output

The script prints a detailed summary to stdout:

```
======================================================================
BALL KNOWER v1.1 BACKTEST
======================================================================
Season: 2025
Training weeks: 1-10 (10 weeks)
Testing weeks: 11-18 (8 weeks)
Edge thresholds: [1, 2, 3, 4]
======================================================================

STEP 1: Calibrating weights on training data
...
======================================================================
CALIBRATED WEIGHTS (v1.1)
======================================================================
  nfelo weight:       0.0324
  substack weight:    0.4069
  epa_off weight:    -2.4325
  epa_def weight:     9.4062
  bias:               2.7577
======================================================================
Training Performance:
  MAE:  1.874 points
  RMSE: 2.119 points
  Games: 140
======================================================================

STEP 2: Generating predictions for test weeks
✓ Generated 120 predictions across 8 weeks

STEP 3: Calculating performance metrics

Model Performance (vs Vegas):
  v1.1 MAE:  2.134 points
  v1.1 RMSE: 2.687 points
  v1.1 Correlation: 0.892

  v1.0 MAE:  2.456 points
  v1.0 RMSE: 3.012 points
  v1.0 Correlation: 0.874

  Games analyzed: 120

======================================================================
STEP 4: Against-The-Spread (ATS) Performance
======================================================================

Edge >= 1 points:
  v1.1: 24-18-2 (57.1% win rate, +8.2% ROI, 44 bets)
  v1.0: 19-22-1 (46.3% win rate, -7.5% ROI, 42 bets)

Edge >= 2 points:
  v1.1: 15-8-1 (65.2% win rate, +22.7% ROI, 24 bets)
  v1.0: 12-11-0 (52.2% win rate, +1.3% ROI, 23 bets)

Edge >= 3 points:
  v1.1: 9-3-0 (75.0% win rate, +45.5% ROI, 12 bets)
  v1.0: 6-6-0 (50.0% win rate, -9.1% ROI, 12 bets)

Edge >= 4 points:
  v1.1: 5-1-0 (83.3% win rate, +63.6% ROI, 6 bets)
  v1.0: 3-3-0 (50.0% win rate, -9.1% ROI, 6 bets)
```

### Key Metrics

#### Model Accuracy
- **MAE (Mean Absolute Error):** Average difference from Vegas lines in points. Lower is better.
  - Good: < 2.0 points
  - Acceptable: 2.0-3.0 points
  - Poor: > 3.0 points

- **RMSE (Root Mean Squared Error):** Similar to MAE but penalizes large errors more heavily.

- **Correlation:** How well the model tracks Vegas lines (0-1). Higher is better.
  - Excellent: > 0.90
  - Good: 0.85-0.90
  - Acceptable: 0.75-0.85

#### ATS Performance
- **Win Rate:** Percentage of bets won (excluding pushes)
  - Break-even at -110 odds: 52.4%
  - Profitable: > 52.4%

- **ROI (Return on Investment):** Total profit/loss as percentage of amount wagered
  - Positive ROI = profitable
  - At -110 odds, 55% win rate ≈ +5% ROI

- **Number of Bets:** Sample size matters. More bets = more confidence in results.
  - Small sample: < 20 bets (high variance)
  - Medium sample: 20-50 bets
  - Large sample: > 50 bets (more reliable)

### CSV Output

The main CSV contains one row per game with columns:

- `week`, `away_team`, `home_team` - Game identification
- `vegas_line` - Vegas spread (negative = home favored)
- `bk_line_v1_1` - Ball Knower v1.1 prediction
- `bk_line_v1_0` - Ball Knower v1.0 prediction (for comparison)
- `edge_v1_1` - v1.1 edge over Vegas (model line - Vegas line)
- `edge_v1_0` - v1.0 edge over Vegas
- `nfelo_diff`, `substack_power_diff`, `epa_off_diff`, `epa_def_diff` - Model components
- `home_score`, `away_score` - Actual scores (if game completed)

**Use this CSV to:**
- Identify high-edge games
- Analyze specific matchups
- Build your own betting strategies
- Compare v1.1 vs v1.0 performance

---

## Example Workflows

### Workflow 1: Add Historical Weeks (1-10) and Run Full Backtest

**Scenario:** You have Week 11 data and just downloaded weeks 1-10.

```bash
# Step 1: Prepare each week's data
for week in {1..10}; do
    python scripts/prepare_weekly_ratings.py \
        --season 2025 \
        --week $week \
        --source-dir raw_downloads/week_${week}/ \
        --nfelo-power power_ratings.csv \
        --nfelo-epa epa_tiers.csv \
        --substack-power substack_ratings.csv \
        --substack-proj weekly_proj.csv
done

# Step 2: Run backtest - train on 1-10, test on 11
python scripts/run_backtest_v1_1.py \
    --season 2025 \
    --train-weeks 1-10 \
    --test-weeks 11 \
    --edge-thresholds 1,2,3,4

# Results saved to:
#   output/backtest_v1_1_2025_train_1-10_test_11.csv
#   output/backtest_v1_1_2025_ats_summary.csv
```

### Workflow 2: Weekly Update - Add New Week and Re-run

**Scenario:** It's now Week 12. You want to add Week 12 data and test on it.

```bash
# Step 1: Prepare Week 12 data
python scripts/prepare_weekly_ratings.py \
    --season 2025 \
    --week 12 \
    --source-dir raw_downloads/ \
    --nfelo-power nfelo_power_ratings_2025_week_12.csv \
    --nfelo-epa nfelo_epa_tiers_2025_week_12.csv \
    --substack-power substack_ratings_week12.csv \
    --substack-proj substack_projections_week12.csv

# Step 2: Run backtest - train on 1-10, test on 11-12
python scripts/run_backtest_v1_1.py \
    --season 2025 \
    --train-weeks 1-10 \
    --test-weeks 11-12 \
    --edge-thresholds 1,2,3,4
```

### Workflow 3: Cross-Validation - Multiple Train/Test Splits

**Scenario:** Test robustness with different train/test splits.

```bash
# Split 1: Train on 1-8, test on 9-11
python scripts/run_backtest_v1_1.py \
    --season 2025 \
    --train-weeks 1-8 \
    --test-weeks 9-11

# Split 2: Train on 1-6,9-11, test on 7-8
python scripts/run_backtest_v1_1.py \
    --season 2025 \
    --train-weeks 1-6,9-11 \
    --test-weeks 7-8

# Split 3: Train on 1-5, test on 6-11
python scripts/run_backtest_v1_1.py \
    --season 2025 \
    --train-weeks 1-5 \
    --test-weeks 6-11
```

Compare results across splits to assess stability.

### Workflow 4: Generate Current Week Predictions

**Scenario:** Use all available historical data to predict the upcoming week.

```bash
# Train on all weeks with results (1-11), predict week 12
python scripts/run_backtest_v1_1.py \
    --season 2025 \
    --train-weeks 1-11 \
    --test-weeks 12 \
    --edge-thresholds 2,3,4

# Output will include predictions even if scores aren't available yet
# Use the CSV to find high-edge games for betting
```

---

## Troubleshooting

### Error: "Schedule file not found"

**Problem:** `data/cache/schedules_2025.csv` is missing.

**Solution:**
- Verify the file exists at this location
- Check that it contains columns: `week`, `home_team`, `away_team`, `spread_line`, `game_type`
- Ensure `game_type == 'REG'` for regular season games

### Error: "Missing ratings for {team}"

**Problem:** Team name mismatch between schedule and ratings files.

**Solution:**
- Check team name normalization in `src/team_mapping.py`
- Verify all teams in the schedule have corresponding entries in power ratings CSVs
- Look for typos or inconsistent abbreviations (e.g., "LA" vs "LAR")

### Warning: "Missing columns in {file}"

**Problem:** Expected columns not found in a data file.

**Solution:**
- Review the validation warning to see which columns are missing
- Check the raw file to confirm column names match expectations
- For EPA files, it may use `'EPA/Play'` instead of `'epa_off'` (this is automatically handled)

### Error: "No valid games found with complete ratings"

**Problem:** No games have all required data sources.

**Solution:**
- Verify you have all three required files for each week:
  - `power_ratings_nfelo_{season}_week_{week}.csv`
  - `epa_tiers_nfelo_{season}_week_{week}.csv`
  - `power_ratings_substack_{season}_week_{week}.csv`
- Check that team names are consistent across files

### Low Sample Size (Few Bets)

**Problem:** At high edge thresholds (e.g., 4+ points), you only get a handful of bets.

**Solution:**
- This is expected - high-edge opportunities are rare
- Use lower thresholds (1-2 points) for more bets, but expect lower win rates
- Combine multiple weeks to increase sample size
- Consider this when interpreting ROI (small samples have high variance)

### v1.1 Not Better Than v1.0

**Problem:** Calibrated weights (v1.1) perform worse than fixed weights (v1.0).

**Possible Causes:**
- **Overfitting:** Training set too small or not representative
- **Data quality:** Missing/incorrect data in training weeks
- **Model assumptions:** Linear model may not capture all patterns

**Solutions:**
- Increase training set size (use more weeks)
- Try different train/test splits
- Review training data for errors or outliers
- Accept that v1.0 may be sufficient if differences are small

---

## Next Steps

Once you're comfortable with the backtesting workflow:

1. **Automate data collection:** Set up scripts to download nfelo/Substack data weekly
2. **Monitor performance:** Track how v1.1 performs on new weeks in real-time
3. **Refine edge thresholds:** Find the optimal balance between volume and accuracy
4. **Explore v1.2+:** Consider adding new features (injuries, weather, etc.) to the model

For model internals, see:
- `ball_knower/models/v1_1_calibration.py` - Implementation
- `ball_knower/models/README.md` - Model documentation
- `notebooks/ball_knower_v1_1_backtest.ipynb` - Interactive analysis

---

**Questions or issues?** Check the repository README or open an issue on GitHub.
