# EPA Data Setup Guide

## The Problem

Network restrictions in this environment prevent automated downloads of NFLverse play-by-play data. However, we can work around this with a hybrid approach.

## The Solution

**Store pre-aggregated team-week statistics instead of full play-by-play**

Benefits:
- ✓ Get all EPA metrics you need
- ✓ Small file size (~100KB vs 500MB+ for full PBP)
- ✓ Git-friendly (can commit to repo)
- ✓ Fast to load and process
- ✓ Easy weekly updates

## Setup Process

### Step 1: Download Play-by-Play Data (One-Time)

On your local machine (not in this environment):

```python
# Install nfl_data_py locally
pip install nfl_data_py

# Download all historical play-by-play
import nfl_data_py as nfl
pbp = nfl.import_pbp_data(range(2009, 2025))

# Save to parquet (compressed)
pbp.to_parquet('pbp_2009_2024_full.parquet')
```

Or download directly from:
https://github.com/nflverse/nflverse-data/releases

### Step 2: Aggregate to Team-Week Stats

Use the `aggregate_pbp_to_team_stats.py` script (provided below) to:
- Process full play-by-play into team-week aggregates
- Calculate offensive/defensive EPA metrics
- Compute success rates, explosive plays, etc.
- Output small CSV file for Git

```bash
python aggregate_pbp_to_team_stats.py pbp_2009_2024_full.parquet
```

This creates: `data/team_week_epa_2009_2024.csv` (~100KB)

### Step 3: Commit Aggregated Data

```bash
git add data/team_week_epa_2009_2024.csv
git commit -m "Add historical team-week EPA data (2009-2024)"
git push
```

### Step 4: Weekly Updates

After each week's games complete:

```python
# Download just current week
pbp_week = nfl.import_pbp_data([2024], week=12)

# Aggregate
python aggregate_pbp_to_team_stats.py pbp_week.parquet --append

# Commit update
git add data/team_week_epa_2009_2024.csv
git commit -m "Update EPA data through Week 12"
git push
```

## Aggregated Data Schema

The team-week CSV will contain:

```
season,week,team,
  # Offensive metrics
  off_plays,off_epa_total,off_epa_per_play,off_success_rate,off_explosive_rate,
  off_pass_epa,off_run_epa,off_third_down_conv_rate,

  # Defensive metrics (EPA allowed to opponent)
  def_plays,def_epa_total,def_epa_per_play,def_success_rate,def_explosive_rate,
  def_pass_epa,def_run_epa,def_third_down_conv_rate,

  # Situational
  redzone_td_rate,turnover_rate,penalty_rate
```

File size: ~100KB (vs 500MB+ for full PBP)

## Alternative: Automated External Pipeline

If you want fully automated weekly updates:

1. Set up a GitHub Action or cron job on a machine with network access
2. Runs weekly: download → aggregate → commit → push
3. This repo pulls the updates automatically

Example GitHub Action:
```yaml
name: Update EPA Data
on:
  schedule:
    - cron: '0 12 * * 2'  # Tuesdays at noon (after MNF)
jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: pip install nfl_data_py pandas
      - run: python scripts/weekly_epa_update.py
      - run: |
          git config user.name "EPA Bot"
          git add data/team_week_epa_*.csv
          git commit -m "Update EPA data - Week $(date +%U)"
          git push
```

## Why This Works

1. **Small Files**: Aggregated data is 100KB vs 500MB raw
2. **Git-Friendly**: Can commit and track changes
3. **Fast**: Loads in milliseconds vs seconds
4. **Complete**: Contains all EPA metrics you need
5. **Maintainable**: Weekly updates are simple
6. **Flexible**: Can re-aggregate from raw if needed

## Next Steps

1. Download raw PBP data locally (one-time)
2. Run aggregation script (I'll create this)
3. Commit aggregated CSV to repo
4. Use in model training (I'll update v1.3 to use this)
5. Set up weekly update process (manual or automated)

This approach gives you all the EPA data you need while working around the network restrictions!
