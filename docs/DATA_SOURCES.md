# Ball Knower Data Sources Reference

**Author:** Ball Knower Team
**Date:** 2024-11-17
**Version:** 1.0.0

This document provides a comprehensive reference for all data sources used in the Ball Knower NFL betting analysis system.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Categories](#data-categories)
3. [Data Providers](#data-providers)
4. [File Naming Convention](#file-naming-convention)
5. [Loading Data](#loading-data)
6. [Data Source Details](#data-source-details)

---

## Overview

The Ball Knower system integrates multiple data sources to generate NFL betting recommendations. Each data source is categorized by type (e.g., power ratings, EPA metrics) and provider (e.g., NFelo, Substack).

### Supported Data Categories

| Category | Description | Providers |
|----------|-------------|-----------|
| `power_ratings` | Team power rankings/ratings | NFelo, Substack |
| `epa_tiers` | EPA per play rankings (offense/defense) | NFelo |
| `strength_of_schedule` | Strength of schedule metrics | NFelo |
| `qb_epa` | Quarterback EPA metrics | Substack |
| `weekly_projections_ppg` | Points-per-game projections | Substack |
| `rest_days` | Days of rest between games | TBD |
| `team_stats` | Team-level statistics | NFLverse |
| `injuries` | Injury reports | TBD |
| `vegas_lines` | Vegas betting lines | TBD |

### Supported Providers

- **NFelo** (`nfelo`): Advanced NFL analytics site with Elo ratings and EPA metrics
- **Substack** (`substack`): NFL analytics newsletter with power ratings and projections
- **NFLverse** (`nflverse`): Open-source NFL data project

---

## File Naming Convention

### Category-First Pattern (Preferred)

**Format:** `{category}_{provider}_{season}_week_{week}.csv`

**Examples:**
- `power_ratings_nfelo_2024_week_11.csv`
- `epa_tiers_nfelo_2024_week_11.csv`
- `qb_epa_substack_2024_week_11.csv`

### Provider-First Pattern (Legacy, Deprecated)

**Format:** `{provider}_{category}_{season}_week_{week}.csv`

**Examples:**
- `nfelo_power_ratings_2024_week_11.csv` ⚠️ **Deprecated**
- `nfelo_epa_tiers_2024_week_11.csv` ⚠️ **Deprecated**

> **Note:** The loader automatically detects both patterns but will issue a deprecation warning for legacy filenames. Please rename to the category-first pattern.

---

## Loading Data

### Using the Unified Loader (Recommended)

```python
from ball_knower.io import loaders

# Load a specific category
power_ratings = loaders.load_power_ratings('nfelo', 2024, 11, './data')

# Load all sources at once
data = loaders.load_all_sources(week=11, season=2024, data_dir='./data')

# Get merged team ratings
merged = loaders.merge_team_ratings(data)
```

### Available Loader Functions

```python
# Category-specific loaders
loaders.load_power_ratings(provider, season, week, data_dir)
loaders.load_epa_tiers(provider, season, week, data_dir)
loaders.load_strength_of_schedule(provider, season, week, data_dir)
loaders.load_qb_epa(provider, season, week, data_dir)
loaders.load_weekly_projections_ppg(provider, season, week, data_dir)
loaders.load_rest_days(provider, season, week, data_dir)
loaders.load_team_stats(provider, season, week, data_dir)
loaders.load_injuries(provider, season, week, data_dir)
loaders.load_vegas_lines(provider, season, week, data_dir)

# Orchestrator functions
loaders.load_all_sources(week, season, data_dir)
loaders.merge_team_ratings(data)
```

---

## Data Source Details

### 1. Power Ratings

**Description:** Overall team strength ratings on various scales.

**Providers:** NFelo, Substack

**NFelo Power Ratings:**
- **Columns:**
  - `team` (str): Team abbreviation
  - `nfelo` (float): NFelo rating (scaled ~1500)
  - `QB Adj` (float): QB adjustment
  - `Value` (float): Value rating
- **Scale:** ~1300-1700 (higher = better)
- **File:** `power_ratings_nfelo_{season}_week_{week}.csv`

**Substack Power Ratings:**
- **Columns:**
  - `team` (str): Team abbreviation
  - `Off.` (float): Offensive rating
  - `Def.` (float): Defensive rating
  - `Ovr.` (float): Overall rating
- **Scale:** ~0-30 (higher = better)
- **File:** `power_ratings_substack_{season}_week_{week}.csv`

---

### 2. EPA Tiers

**Description:** Expected Points Added per play rankings.

**Providers:** NFelo

**Columns:**
- `team` (str): Team abbreviation
- `epa_off` (float): Offensive EPA per play
- `epa_def` (float): Defensive EPA per play (lower = better defense)
- `epa_margin` (float): epa_off - epa_def

**Scale:** ~-0.3 to +0.3 per play

**File:** `epa_tiers_nfelo_{season}_week_{week}.csv`

---

### 3. Strength of Schedule

**Description:** Difficulty of opponents played/upcoming.

**Providers:** NFelo

**Columns:**
- `team` (str): Team abbreviation
- `sos_to_date` (float): SOS for games played
- `sos_remaining` (float): SOS for games remaining
- `sos_total` (float): SOS for full season

**Scale:** ~-5 to +5 (higher = harder schedule)

**File:** `strength_of_schedule_nfelo_{season}_week_{week}.csv`

---

### 4. QB EPA

**Description:** Quarterback-level EPA metrics.

**Providers:** Substack

**Columns:**
- `QB` (str): Quarterback name
- `team` (str): Team abbreviation
- `EPA_per_play` (float): QB's EPA per play
- `completions` (int): Completions
- `attempts` (int): Attempts
- `pass_yards` (int): Passing yards
- `pass_td` (int): Passing TDs

**File:** `qb_epa_substack_{season}_week_{week}.csv`

---

### 5. Weekly Projections (PPG)

**Description:** Points-per-game projections for upcoming games.

**Providers:** Substack

**Columns:**
- `Matchup` (str): "Team1 at Team2" format
- `team_home` (str): Home team abbreviation
- `team_away` (str): Away team abbreviation
- `home_ppg` (float): Home team projected PPG
- `away_ppg` (float): Away team projected PPG
- `substack_spread_line` (float): Projected spread

**File:** `weekly_projections_ppg_substack_{season}_week_{week}.csv`

---

### 6. Vegas Lines

**Description:** Current betting lines from sportsbooks.

**Providers:** Various

**Columns:**
- `game_id` (str): Unique game identifier
- `home_team` (str): Home team abbreviation
- `away_team` (str): Away team abbreviation
- `spread` (float): Point spread (negative = home favored)
- `total` (float): Over/under total points
- `home_moneyline` (int): Moneyline for home team
- `away_moneyline` (int): Moneyline for away team

**File:** `vegas_lines_{provider}_{season}_week_{week}.csv`

---

### 7. Team Stats

**Description:** Aggregated team-level statistics.

**Providers:** NFLverse

**Common Columns:**
- `team` (str): Team abbreviation
- `season` (int): Season year
- `week` (int): Week number
- `completions` (int): Pass completions
- `attempts` (int): Pass attempts
- `passing_yards` (int): Passing yards
- `rushing_yards` (int): Rushing yards
- `turnovers` (int): Turnovers
- ... (many more statistical categories)

**File:** `team_stats_nflverse_{season}_week_{week}.csv`

---

### 8. Injuries

**Description:** Player injury status and impact.

**Status:** To be implemented

**Expected Columns:**
- `player_name` (str): Player name
- `team` (str): Team abbreviation
- `position` (str): Position
- `injury_status` (str): Out/Doubtful/Questionable/Probable
- `injury_description` (str): Body part/injury type

**File:** `injuries_{provider}_{season}_week_{week}.csv`

---

### 9. Rest Days

**Description:** Days of rest between games (e.g., Thursday night games).

**Status:** To be implemented

**Expected Columns:**
- `team` (str): Team abbreviation
- `week` (int): Week number
- `rest_days` (int): Days since previous game

**File:** `rest_days_{provider}_{season}_week_{week}.csv`

---

## Team Name Normalization

All team names are automatically normalized to standard abbreviations using `src/team_mapping.py`:

| Common Variants | Standard Abbreviation |
|-----------------|----------------------|
| `Los Angeles Rams`, `LA Rams`, `LAR` | `LAR` |
| `Los Angeles Chargers`, `LA Chargers`, `LAC` | `LAC` |
| `Kansas City`, `KC`, `KAN` | `KC` |
| `New England`, `NE`, `NEP` | `NE` |
| ... | ... |

The loader automatically applies normalization to all `team` columns in loaded data.

---

## Error Handling

The unified loader handles missing data gracefully:

1. **Missing Files:** Returns `None` and issues a warning (for required categories) or silent fallback (for optional categories).
2. **Legacy Filenames:** Loads successfully but issues a `DeprecationWarning`.
3. **Invalid Data:** Raises descriptive errors for malformed CSVs.

### Example Warning Output

```
/home/user/BK_Build/ball_knower/io/loaders.py:120: DeprecationWarning:
Using legacy filename pattern: nfelo_power_ratings_2024_week_11.csv.
Please rename to category-first: power_ratings_nfelo_2024_week_11.csv
```

---

## Future Enhancements

1. **Remote Data Fetching:** Automatic download from NFelo/Substack APIs
2. **Caching:** Smart caching to avoid redundant loads
3. **Validation:** Schema validation for loaded data
4. **Live Updates:** Real-time injury/line updates during gameday

---

## Support

For questions or issues with data loading:

1. Check file naming matches the category-first convention
2. Verify data files exist in the `data_dir` directory
3. Review warnings/errors in console output
4. Consult `docs/DATA_MIGRATION_SUMMARY.md` for migration guidance

---

**Last Updated:** 2024-11-17
**Version:** 1.0.0
