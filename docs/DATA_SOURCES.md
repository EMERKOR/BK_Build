# Data Sources Documentation

[Docs Home](README.md) | [Architecture](ARCHITECTURE_OVERVIEW.md) | [Data Sources](DATA_SOURCES.md) | [Feature Tiers](FEATURE_TIERS.md) | [Spec](BALL_KNOWER_SPEC.md) | [Dev Guide](DEVELOPMENT_GUIDE.md)

---

## Overview

This document describes the data sources used by Ball Knower, their naming conventions, directory structure, and how they are loaded and validated.

---

## Canonical Naming Convention

### Category-First Structure

The **canonical naming convention** is category-first:

```
{category}_{provider}_{season}_week_{week}.csv
```

**Examples**:
- `power_ratings_nfelo_2025_week_11.csv`
- `qb_epa_substack_2025_week_11.csv`
- `odds_fivethirtyeight_2025_week_11.csv`

### Provider-First Fallback

For backward compatibility, loaders also support provider-first naming:

```
{provider}_{category}_{season}_week_{week}.csv
```

**Examples**:
- `nfelo_power_ratings_2025_week_11.csv`
- `substack_qb_epa_2025_week_11.csv`

The unified loader (`ball_knower/io/loaders/`) tries category-first, then falls back to provider-first if the file is not found.

---

## Supported Data Providers

### 1. nfelo

**Source**: [nfeloapp.com](https://www.nfeloapp.com)

**Description**: Elo-based power ratings and historical game data

**Categories**:
- `power_ratings` — Team Elo ratings
- `epa_tiers` — EPA offensive/defensive metrics
- `strength_of_schedule` — SOS ratings

**Historical Data**:
- `nfelo_games.csv` — Historical Elo ratings and Vegas lines (2009-present)

### 2. Substack

**Source**: Various Substack-based NFL analysts

**Description**: Power ratings, QB metrics, and weekly projections

**Categories**:
- `power_ratings` — Team power rankings
- `qb_epa` — QB-level EPA and performance stats
- `weekly_projections_ppg` — Game-level projections and spreads

### 3. nflverse

**Source**: [nfl_data_py](https://github.com/cooperdff/nfl_data_py) Python package

**Description**: Official NFL statistics and schedules

**Categories**:
- `schedules` — Game schedules and results
- `games` — Game-level metadata
- `play_by_play` — Play-by-play data (for EPA calculations)
- `team_stats` — Team-level offensive/defensive stats

**Coverage**: 1999-present

---

## Directory Structure

### Current Season Data

```
data/
├── elo/
│   ├── nfelo/
│   │   └── power_ratings_nfelo_2025_week_11.csv
│   └── substack/
│       └── power_ratings_substack_2025_week_11.csv
├── odds/
│   └── fivethirtyeight/
│       └── odds_fivethirtyeight_2025_week_11.csv
├── stats/
│   └── nflverse/
│       └── schedules.parquet
└── reference/
    └── team_abbreviations.csv
```

### Historical Data

```
data/
└── historical/
    ├── nfelo_games.csv
    ├── schedules_1999_2024.parquet
    └── games_metadata.csv
```

---

## Data Validation Checklist

When adding new data sources, ensure:

- [ ] File follows category-first naming convention
- [ ] Provider directory exists under category
- [ ] Team names are normalized to standard abbreviations
- [ ] Date/time fields are in ISO format (`YYYY-MM-DD`)
- [ ] Required columns are present (see schemas below)
- [ ] Missing values are handled appropriately
- [ ] No post-game information leakage (for predictive features)

---

## Loader Workflow

### Category-First ∨ Provider-First Fallback

The unified loader handles both naming conventions automatically:

```python
# Pseudocode for loader logic
def load_data(category, provider, season, week):
    # Try category-first
    path = f"data/{category}/{provider}/{category}_{provider}_{season}_week_{week}.csv"
    if file_exists(path):
        return read_csv(path)

    # Fallback to provider-first
    path = f"data/{category}/{provider}/{provider}_{category}_{season}_week_{week}.csv"
    if file_exists(path):
        return read_csv(path)

    # Not found
    raise FileNotFoundError(f"Data file not found for {category}/{provider}")
```

### Validation on Load

All loaders perform validation:

1. Check required columns exist
2. Normalize team names
3. Convert date/time formats
4. Handle missing values
5. Validate ranges (e.g., EPA values, ratings)

---

## Pseudo-Schema Examples

### Power Ratings

```python
{
    "team": str,           # Standard team abbreviation (e.g., "KC", "LAR")
    "elo_rating": float,   # Current Elo rating
    "qb_adj": float,       # QB adjustment to Elo
    "overall_rating": float # Combined power rating
}
```

### QB Metrics

```python
{
    "team_abbr": str,       # Team abbreviation
    "qb_name": str,         # Quarterback name
    "epa_per_play": float,  # EPA per play (passing)
    "completions": int,     # Completions
    "attempts": int         # Attempts
}
```

### Weekly Projections

```python
{
    "matchup": str,          # "Team A @ Team B"
    "home_team": str,        # Home team abbreviation
    "away_team": str,        # Away team abbreviation
    "projected_spread": float, # Projected spread (home team perspective)
    "projected_total": float   # Projected total points
}
```

### Historical Games

```python
{
    "game_id": str,          # Unique game identifier
    "season": int,           # Season year
    "week": int,             # Week number
    "home_team": str,        # Home team abbreviation
    "away_team": str,        # Away team abbreviation
    "home_score": int,       # Final home score
    "away_score": int,       # Final away score
    "spread_line": float,    # Vegas closing spread
    "total_line": float      # Over/under total
}
```

---

## Team Abbreviation Rules

### Placeholder Section

[To be documented]

- Standard abbreviation mapping (e.g., "Kansas City Chiefs" → "KC")
- Provider-specific quirks and normalization rules
- Handling of team relocations (e.g., "STL" → "LAR")
- Historical team name changes

---

## Adding New Data Sources

### Checklist

When adding a new data provider:

1. Create provider directory under appropriate category
2. Follow category-first naming convention
3. Document provider in this file
4. Add loader function to `ball_knower/io/loaders/`
5. Define schema validation rules
6. Add unit tests with fixtures
7. Update [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)

---

## References

- [FEATURE_TIERS.md](FEATURE_TIERS.md) — Feature organization and leakage prevention
- [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) — System architecture
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) — Adding new loaders

---

**Status**: This document is a living reference and will be updated as new data sources are integrated.
