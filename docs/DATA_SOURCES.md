# Ball Knower Data Sources

## Overview

Ball Knower uses data from multiple NFL analytics sources to generate betting predictions. This document describes the data sources, file naming conventions, and available features.

## Data Directory Structure

```
data/
├── current_season/     # Current week data files (2025 Week 11)
└── reference/          # Static reference data
```

## File Naming Convention

### Target Convention (Category-First)

**Format:** `{category}_{provider}_{season}_week_{week}.csv`

**Examples:**
- `power_ratings_nfelo_2025_week_11.csv`
- `epa_tiers_nfelo_2025_week_11.csv`
- `qb_rankings_nfelo_2025_week_11.csv`
- `power_ratings_substack_2025_week_11.csv`
- `qb_epa_substack_2025_week_11.csv`

### Legacy Convention (Provider-First)

**Format:** `{provider}_{category_variant}_{season}_week_{week}.csv`

**Current files (legacy naming):**
- `nfelo_power_ratings_2025_week_11.csv`
- `nfelo_epa_tiers_off_def_2025_week_11.csv`
- `nfelo_qb_rankings_2025_week_11.csv`
- `nfelo_strength_of_schedule_2025_week_11.csv`
- `substack_power_ratings_2025_week_11.csv`
- `substack_qb_epa_2025_week_11.csv`
- `substack_weekly_proj_ppg_2025_week_11.csv`
- `substack_weekly_proj_elo_2025_week_11.csv`

**Note:** The unified loader (`ball_knower.io.loaders`) automatically handles both naming conventions with fallback support. Legacy filenames will trigger deprecation warnings.

## Data Categories

### 1. Power Ratings

**Providers:** nfelo, substack

**Description:** Overall team strength ratings combining offensive/defensive performance.

**nfelo Power Ratings Fields:**
- `team`: Team abbreviation (normalized)
- `nfelo`: Main ELO rating
- `QB Adj`: Quarterback adjustment
- `Value`: Overall value metric
- `WoW`: Week-over-week change
- `YTD`: Year-to-date performance

**Substack Power Ratings Fields:**
- `team`: Team abbreviation (normalized)
- `Off.`: Offensive rating
- `Def.`: Defensive rating
- `Ovr.`: Overall rating

### 2. EPA Tiers

**Providers:** nfelo

**Description:** Expected Points Added (EPA) per play for offense and defense.

**Fields:**
- `team`: Team abbreviation (normalized)
- `epa_off`: Offensive EPA per play
- `epa_def`: Defensive EPA per play (opponent's EPA against)
- `epa_margin`: Difference between offensive and defensive EPA

### 3. Strength of Schedule (SOS)

**Providers:** nfelo

**Description:** Metrics quantifying difficulty of past and future opponents.

**Fields:**
- `team`: Team abbreviation (normalized)
- Various SOS metrics for remaining/played games

### 4. QB Rankings

**Providers:** nfelo

**Description:** Quarterback performance rankings.

**Fields:**
- `team`: Team abbreviation (normalized)
- QB-specific performance metrics

### 5. QB EPA

**Providers:** substack

**Description:** Quarterback-level EPA metrics.

**Fields:**
- `team`: Team abbreviation (normalized, from primary team)
- `Tms`: Original team codes (may include multiple teams)
- QB EPA metrics

### 6. Weekly Projections (PPG)

**Providers:** substack

**Description:** Points-per-game based game projections and spreads.

**Fields:**
- `Matchup`: Game matchup string
- `team_away`: Normalized away team abbreviation
- `team_home`: Normalized home team abbreviation
- `substack_spread_line`: Projected spread
- PPG projections

### 7. Weekly Projections (ELO)

**Providers:** substack

**Description:** ELO-based game projections and spreads.

**Fields:**
- Similar to PPG projections but ELO-derived

### 8. Win Totals

**Providers:** nfelo

**Description:** Season win total projections and betting lines.

**Fields:**
- `team`: Team abbreviation (normalized)
- Win total projections

### 9. Receiving Leaders

**Providers:** nfelo

**Description:** Top receivers and receiving metrics.

**Fields:**
- Receiver names, teams, stats

## Data Providers

### nfelo (https://nfelo.com)

**Data Types:**
- Power ratings (ELO-based)
- EPA tiers
- Strength of schedule
- QB rankings
- Win totals
- Receiving leaders

**Update Frequency:** Weekly during NFL season

**Format:** CSV with single header row

### Substack (nflfastr)

**Data Types:**
- Power ratings
- QB EPA
- Weekly game projections (PPG and ELO)

**Update Frequency:** Weekly during NFL season

**Format:** CSV with double header rows (first row skipped during load)

## Loading Data

### Using Unified Loaders (Recommended)

```python
from ball_knower.io import loaders

# Load specific category
nfelo_power = loaders.load_power_ratings("nfelo", season=2025, week=11)
substack_power = loaders.load_power_ratings("substack", season=2025, week=11)

# Load all sources
data = loaders.load_all_sources(week=11, season=2025)

# Access merged ratings
merged_ratings = data["merged_ratings"]
```

### Using Legacy Interface (Deprecated)

```python
from src import data_loader

# Load specific datasets (will show deprecation warnings)
nfelo_power = data_loader.load_nfelo_power_ratings()
substack_power = data_loader.load_substack_power_ratings()

# Load all data
data = data_loader.load_all_current_week_data()
merged = data_loader.merge_current_week_ratings()
```

## Team Name Normalization

All loaders automatically normalize team names to standard 2-3 character abbreviations using `src.team_mapping`:

**Normalization Examples:**
- "Los Angeles Rams" → "LAR"
- "Rams" → "LAR"
- "ram" → "LAR"
- "Kansas City" → "KC"
- "Chiefs" → "KC"

**Standard Abbreviations:**
```
ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN,
DET, GB, HOU, IND, JAX, KC, LAC, LAR, LV, MIA,
MIN, NE, NO, NYG, NYJ, PHI, PIT, SEA, SF, TB,
TEN, WAS
```

## Merged Ratings Structure

The `merged_ratings` DataFrame combines all team-level metrics:

**Columns:**
- `team`: Team abbreviation
- `nfelo`: nfelo power rating
- `QB Adj`: QB adjustment
- `Value`: Overall value
- `epa_off`: Offensive EPA/play
- `epa_def`: Defensive EPA/play
- `epa_margin`: EPA differential
- `Off.`: Substack offensive rating
- `Def.`: Substack defensive rating
- `Ovr.`: Substack overall rating

**Shape:** 32 teams × 10 features

## Data Update Process

1. Download latest CSV files from providers (weekly)
2. Save to `data/current_season/` with appropriate naming
3. Update `src/config.py` with current season/week if needed
4. Run loaders to validate data

## Migration Status

**Current State (2025-11-17):**
- ✅ Phase 1: Unified loaders created (`ball_knower.io.loaders`)
- ✅ Phase 2: Compatibility layer added (`src/data_loader.py`)
- ⏳ Phase 3: Scripts still use legacy imports (pending)
- ⏳ Phase 4: CSV files still use provider-first naming (pending)
- ⏳ Phase 5: Legacy code cleanup (pending)

**Files use legacy provider-first naming** with automatic fallback support.

**Recommended:** Start using `ball_knower.io.loaders` for new code.

---

**Last Updated:** 2025-11-17
**Current Season:** 2025 Week 11
**Data Sources:** nfelo, Substack (nflfastr)
