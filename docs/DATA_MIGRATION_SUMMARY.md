# Data Loading Migration Summary

This document summarizes the migration to the unified naming convention defined in `DATA_SOURCES.md`.

---

## New Unified Module

Created: `ball_knower/io/loaders.py`

This module provides:
- Strict enforcement of the `<category>_<source>_<year>_week_<week>.csv` naming convention
- Category-specific loaders: `load_power_ratings()`, `load_team_epa()`, `load_qb_metrics()`, `load_schedule_context()`
- Generic loader: `load_weekly_file(category, source, year, week)`
- Historical data loader: `load_historical_file(category, source, season)`
- Multi-source loading: `load_all_sources(category, year, week, sources)`
- Validation tools: `validate_naming_convention()`, `print_validation_report()`
- Legacy compatibility layer for gradual migration

---

## Old Code Patterns Replaced

### 1. Old Pattern: Provider-First Naming in src/config.py

**OLD (lines 30-42):**
```python
NFELO_POWER_RATINGS = CURRENT_SEASON_DIR / 'nfelo_power_ratings_2025_week_11.csv'
NFELO_SOS = CURRENT_SEASON_DIR / 'nfelo_strength_of_schedule_2025_week_11.csv'
NFELO_EPA_TIERS = CURRENT_SEASON_DIR / 'nfelo_epa_tiers_off_def_2025_week_11.csv'
NFELO_QB_RANKINGS = CURRENT_SEASON_DIR / 'nfelo_qb_rankings_2025_week_11.csv'
SUBSTACK_POWER_RATINGS = CURRENT_SEASON_DIR / 'substack_power_ratings_2025_week_11.csv'
SUBSTACK_QB_EPA = CURRENT_SEASON_DIR / 'substack_qb_epa_2025_week_11.csv'
```

**NEW:**
```python
from ball_knower.io.loaders import load_power_ratings, load_team_epa, load_qb_metrics

# Load power ratings from any source
nfelo_ratings = load_power_ratings('nfelo', 2025, 11)
substack_ratings = load_power_ratings('substack', 2025, 11)

# Load EPA data
team_epa = load_team_epa('nfelo', 2025, 11)

# Load QB metrics
qb_data = load_qb_metrics('nfelo', 2025, 11)
```

---

### 2. Old Pattern: Function Per Provider in src/data_loader.py

**OLD (lines 108-246):**
```python
def load_nfelo_power_ratings():
    df = pd.read_csv(NFELO_POWER_RATINGS)
    # ... normalization code
    return df

def load_nfelo_epa_tiers():
    df = pd.read_csv(NFELO_EPA_TIERS)
    # ... normalization code
    return df

def load_substack_power_ratings():
    df = pd.read_csv(SUBSTACK_POWER_RATINGS)
    # ... normalization code
    return df
```

**NEW:**
```python
from ball_knower.io.loaders import load_power_ratings, load_team_epa

# Single pattern for all providers
nfelo_power = load_power_ratings('nfelo', 2025, 11)
substack_power = load_power_ratings('substack', 2025, 11)
fivethirtyeight_power = load_power_ratings('fivethirtyeight', 2025, 11)

# Or load all at once
all_ratings = load_all_sources('power_ratings', 2025, 11, sources=['nfelo', 'substack'])
```

---

### 3. Old Pattern: Hardcoded Paths

**OLD:**
```python
df = pd.read_csv('data/current_season/nfelo_power_ratings_2025_week_11.csv')
```

**NEW:**
```python
from ball_knower.io.loaders import load_power_ratings
df = load_power_ratings('nfelo', 2025, 11)
```

---

## Required File Renames

The following files in `data/current_season/` need to be renamed to match the new convention:

### Core Data Files (Required for Ball Knower)

| Current Filename | New Filename | Category | Source |
|-----------------|--------------|----------|---------|
| `nfelo_power_ratings_2025_week_11.csv` | `power_ratings_nfelo_2025_week_11.csv` | power_ratings | nfelo |
| `nfelo_epa_tiers_off_def_2025_week_11.csv` | `team_epa_nfelo_2025_week_11.csv` | team_epa | nfelo |
| `nfelo_qb_rankings_2025_week_11.csv` | `qb_metrics_nfelo_2025_week_11.csv` | qb_metrics | nfelo |
| `nfelo_strength_of_schedule_2025_week_11.csv` | `schedule_context_nfelo_2025_week_11.csv` | schedule_context | nfelo |
| `substack_power_ratings_2025_week_11.csv` | `power_ratings_substack_2025_week_11.csv` | power_ratings | substack |
| `substack_qb_epa_2025_week_11.csv` | `qb_metrics_substack_2025_week_11.csv` | qb_metrics | substack |

### Supplementary Files (May be archived or categorized differently)

| Current Filename | Action | Notes |
|-----------------|---------|-------|
| `nfelo_nfl_receiving_leaders_2025_week_11.csv` | Archive or categorize as `player_stats_nfelo_*` | Player-level data, not team-level |
| `nfelo_nfl_win_totals_2025_week_11 (1).csv` | Rename to remove `(1)`, categorize as `schedule_context_nfelo_*` | Fix duplicate naming |
| `substack_weekly_proj_elo_2025_week_11.csv` | Categorize as `power_ratings_substack_proj_2025_week_11.csv` | Projection variant of power ratings |
| `substack_weekly_proj_ppg_2025_week_11.csv` | Categorize as `power_ratings_substack_proj_2025_week_11.csv` | Projection variant of power ratings |

---

## Bash Script for Renaming

Run this script from the project root to rename files:

```bash
#!/bin/bash
cd data/current_season/

# Core renames
mv nfelo_power_ratings_2025_week_11.csv power_ratings_nfelo_2025_week_11.csv
mv nfelo_epa_tiers_off_def_2025_week_11.csv team_epa_nfelo_2025_week_11.csv
mv nfelo_qb_rankings_2025_week_11.csv qb_metrics_nfelo_2025_week_11.csv
mv nfelo_strength_of_schedule_2025_week_11.csv schedule_context_nfelo_2025_week_11.csv
mv substack_power_ratings_2025_week_11.csv power_ratings_substack_2025_week_11.csv
mv substack_qb_epa_2025_week_11.csv qb_metrics_substack_2025_week_11.csv

echo "✓ Core files renamed"
```

---

## Code Files Requiring Updates

After renaming data files, the following code files need to import the new loader module:

### High Priority (Active Production Code)
- `ball_knower_v1_final.py` - Line 22: Replace `from src import data_loader` with new loaders
- `bk_v1_final.py` - Update data loading calls
- `predict_current_week.py` - Update data loading calls
- `run_demo.py` - Update data loading calls

### Medium Priority (Utilities)
- `src/data_loader.py` - Mark as deprecated, redirect to `ball_knower.io.loaders`
- `src/config.py` - Remove hardcoded file paths (lines 30-42), use loader functions instead

### Low Priority (Archived/Experimental)
- Files in `archive/v1_experiments/` - Can remain unchanged (historical reference)
- Files in `archive/v2_experiments/` - Can remain unchanged

---

## Migration Strategy

### Phase 1: Create New Loaders ✓ COMPLETED
- [x] Create `ball_knower/io/loaders.py`
- [x] Implement category-specific loaders
- [x] Add validation and error messages
- [x] Include legacy compatibility layer

### Phase 2: Rename Data Files (NEXT STEP)
- [ ] Rename core data files in `data/current_season/`
- [ ] Run validation: `python -c "from ball_knower.io.loaders import print_validation_report; print_validation_report()"`
- [ ] Verify all core files are recognized

### Phase 3: Update Active Code (TODO)
- [ ] Update `ball_knower_v1_final.py`
- [ ] Update `bk_v1_final.py`
- [ ] Update `predict_current_week.py`
- [ ] Update `run_demo.py`
- [ ] Test predictions still work

### Phase 4: Deprecate Old Code (TODO)
- [ ] Add deprecation warnings to `src/data_loader.py`
- [ ] Update `src/config.py` to use new loaders
- [ ] Remove hardcoded file paths

### Phase 5: Documentation (TODO)
- [ ] Update README.md with new loading examples
- [ ] Create data loading tutorial notebook

---

## Benefits of New System

1. **Consistency**: All data follows the same `<category>_<source>_<year>_week_<week>` pattern
2. **Discoverability**: Easy to find all power ratings: `power_ratings_*.csv`
3. **Extensibility**: Adding new providers (e.g., PFF, ESPN) requires no code changes
4. **Validation**: Automatic validation catches naming errors
5. **Documentation**: Self-documenting filenames encode category, source, and time period
6. **Type Safety**: Category-specific loaders prevent mixing incompatible data types

---

## Example Usage

### Load Single Source
```python
from ball_knower.io.loaders import load_power_ratings, load_team_epa, load_qb_metrics

# Load week 11 data
nfelo_ratings = load_power_ratings('nfelo', 2025, 11)
team_epa = load_team_epa('nfelo', 2025, 11)
qb_data = load_qb_metrics('substack', 2025, 11)
```

### Load Multiple Sources
```python
from ball_knower.io.loaders import load_all_sources

# Load power ratings from all available sources
all_ratings = load_all_sources('power_ratings', 2025, 11)
nfelo_df = all_ratings['nfelo']
substack_df = all_ratings['substack']
```

### Load Historical Data
```python
from ball_knower.io.loaders import load_historical_file

# Load full season historical data
historical_epa = load_historical_file('team_epa', 'nflverse', 2024)
```

### Validate File Naming
```python
from ball_knower.io.loaders import print_validation_report

# Check if all files follow convention
print_validation_report()
```

---

## Questions or Issues?

See:
- `docs/DATA_SOURCES.md` - Naming convention specification
- `ball_knower/io/loaders.py` - Source code with detailed docstrings
- `docs/BALL_KNOWER_SPEC.md` - Overall model architecture
