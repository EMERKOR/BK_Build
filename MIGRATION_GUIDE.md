# Ball Knower Migration Guide

This guide helps you migrate code from the old ad-hoc data loading system to the new unified loader and category-first naming convention.

---

## Overview

### Old World (Pre-v1.2)
- **File naming**: Provider-first (e.g., `nfelo_power_ratings_2025_week_11.csv`)
- **Data loading**: Ad-hoc `pd.read_csv()` calls scattered throughout scripts
- **Dependencies**: Heavy reliance on `src.data_loader` module
- **Flexibility**: Low - hardcoded paths and filenames

### New World (v1.2+)
- **File naming**: Category-first (e.g., `power_ratings_nfelo_2025_week_11.csv`)
- **Data loading**: Centralized `ball_knower.io.loaders` module
- **Configuration**: Single source of truth in `src.config` (`CURRENT_SEASON`, `CURRENT_WEEK`)
- **Flexibility**: High - automatic fallback to legacy naming, provider-agnostic API

---

## File Naming Migration

### Naming Convention

**New format (preferred):**
```
{category}_{provider}_{season}_week_{week}.csv
```

**Legacy format (deprecated, but still supported):**
```
{provider}_{category}_{season}_week_{week}.csv
```

### Category-First Examples

Here are the actual files we have migrated to category-first naming:

| Category | Provider | Filename |
|----------|----------|----------|
| Power Ratings | nfelo | `power_ratings_nfelo_2025_week_11.csv` |
| Power Ratings | substack | `power_ratings_substack_2025_week_11.csv` |
| QB EPA | substack | `qb_epa_substack_2025_week_11.csv` |
| Weekly Projections PPG | substack | `weekly_projections_ppg_substack_2025_week_11.csv` |
| EPA Tiers | nfelo | `epa_tiers_nfelo_2025_week_11.csv` |
| Strength of Schedule | nfelo | `strength_of_schedule_nfelo_2025_week_11.csv` |

### Still Using Legacy Names (to be migrated)

These files are still in provider-first format and should be renamed when convenient:

| Old Name (Provider-First) | New Name (Category-First) |
|---------------------------|---------------------------|
| `nfelo_qb_rankings_2025_week_11.csv` | `qb_rankings_nfelo_2025_week_11.csv` |
| `nfelo_nfl_receiving_leaders_2025_week_11.csv` | `nfl_receiving_leaders_nfelo_2025_week_11.csv` |
| `nfelo_nfl_win_totals_2025_week_11 (1).csv` | `nfl_win_totals_nfelo_2025_week_11.csv` |
| `substack_weekly_proj_elo_2025_week_11.csv` | `weekly_projections_elo_substack_2025_week_11.csv` |

### Automatic Fallback

**Good news**: You don't need to rename everything at once! The unified loader automatically tries category-first naming first, then falls back to provider-first naming if needed.

```python
# This works regardless of whether the file is named:
#   - power_ratings_nfelo_2025_week_11.csv (new)
#   - nfelo_power_ratings_2025_week_11.csv (old)
from ball_knower.io import loaders
df = loaders.load_power_ratings(provider="nfelo", season=2025, week=11)
```

---

## Code Migration: Data Loading

### Migration Pattern 1: Simple pd.read_csv → Unified Loader

**OLD (direct CSV reading):**
```python
import pandas as pd
from src import config

# Hardcoded path and filename
df = pd.read_csv("data/current_season/nfelo_power_ratings_2025_week_11.csv")
```

**NEW (unified loader):**
```python
from ball_knower.io import loaders
from src import config

# Loader resolves the filename automatically
df = loaders.load_power_ratings(
    provider="nfelo",
    season=config.CURRENT_SEASON,
    week=config.CURRENT_WEEK
)
```

**Benefits:**
- No hardcoded paths or filenames
- Automatic team name normalization
- Works with both old and new file naming conventions
- Handles multi-row headers (common in Substack files)

### Migration Pattern 2: src.data_loader → Unified Loader

**OLD (using src.data_loader):**
```python
from src import data_loader

power_ratings = data_loader.load_nfelo_power_ratings()
qb_epa = data_loader.load_substack_qb_epa()
```

**NEW (unified loader):**
```python
from ball_knower.io import loaders
from src import config

power_ratings = loaders.load_power_ratings(
    provider="nfelo",
    season=config.CURRENT_SEASON,
    week=config.CURRENT_WEEK
)

qb_epa = loaders.load_qb_epa(
    provider="substack",
    season=config.CURRENT_SEASON,
    week=config.CURRENT_WEEK
)
```

### Migration Pattern 3: Loading All Sources

**OLD (multiple manual loads):**
```python
from src import data_loader

power_nfelo = data_loader.load_nfelo_power_ratings()
power_substack = data_loader.load_substack_power_ratings()
qb_epa = data_loader.load_substack_qb_epa()
# ... more manual loads
```

**NEW (single orchestrator call):**
```python
from ball_knower.io import loaders
from src import config

# Load everything at once
all_data = loaders.load_all_sources(
    season=config.CURRENT_SEASON,
    week=config.CURRENT_WEEK
)

# Access individual DataFrames
power_nfelo = all_data['power_ratings_nfelo']
power_substack = all_data['power_ratings_substack']
qb_epa = all_data['qb_epa_substack']

# Or use the pre-merged ratings
merged_ratings = all_data['merged_ratings']
```

**Available keys in `load_all_sources()` result:**
- `power_ratings_nfelo`
- `epa_tiers_nfelo`
- `strength_of_schedule_nfelo`
- `power_ratings_substack`
- `qb_epa_substack`
- `weekly_projections_ppg_substack`
- `merged_ratings` (all sources merged on 'team' column)

---

## Available Loader Functions

The `ball_knower.io.loaders` module provides:

| Function | Parameters | Returns |
|----------|------------|---------|
| `load_power_ratings()` | `provider`, `season`, `week`, `data_dir` (optional) | Power ratings DataFrame |
| `load_epa_tiers()` | `provider`, `season`, `week`, `data_dir` (optional) | EPA tiers DataFrame |
| `load_strength_of_schedule()` | `provider`, `season`, `week`, `data_dir` (optional) | Strength of schedule DataFrame |
| `load_qb_epa()` | `provider`, `season`, `week`, `data_dir` (optional) | QB EPA DataFrame |
| `load_weekly_projections_ppg()` | `provider`, `season`, `week`, `data_dir` (optional) | Weekly projections DataFrame |
| `load_all_sources()` | `season`, `week`, `data_dir` (optional) | Dict of all DataFrames + merged ratings |

**Common providers:** `"nfelo"`, `"substack"`

---

## Recommended Scripts vs Legacy Scripts

### Current, Supported Entry Points

These scripts follow the new unified loader spec and are actively maintained:

- **`run_demo.py`** - Main demo script showing Ball Knower predictions
- **`predict_current_week.py`** - Week 11 predictions with v1.2 model and betting analytics

**Use these scripts as examples** when writing new code.

### Legacy / Experimental Scripts

The following scripts predate the unified loader system and may not follow the current spec:

- `calibrate_regression.py`
- `calibrate_simple.py`
- `ball_knower_v1_final.py`
- `ball_knower_v1_1.py`
- `bk_v1_1_with_adjustments.py`
- `calibrate_to_vegas.py`
- `investigate_data.py`

**Status:** These scripts may still work via the compatibility shim (`src/data_loader.py`), but are not guaranteed to be maintained. They may be refactored or removed in future versions.

**If you're using legacy scripts:** Consider migrating them to the new loader API using the patterns shown above.

---

## Tips for Adding New Data Sources

### 1. File Naming

Use the category-first convention:
```
{category}_{provider}_{season}_week_{week}.csv
```

**Example:** If you're adding a new "defensive_rankings" category from "pff" provider:
```
defensive_rankings_pff_2025_week_12.csv
```

### 2. File Location

- **Current week data:** `data/current_season/`
- **Reference data:** `data/reference/`
- **Historical archives:** `data/archive/` (if needed)

### 3. Using the Loader

If your category isn't built into the loader yet, you can still use `_resolve_file()` for automatic fallback:

```python
from ball_knower.io.loaders import _resolve_file
import pandas as pd

# This will automatically try both category-first and provider-first
file_path = _resolve_file(
    category="defensive_rankings",
    provider="pff",
    season=2025,
    week=12
)
df = pd.read_csv(file_path)
```

### 4. Adding New Loader Functions

If you want to add a dedicated loader function (e.g., `load_defensive_rankings()`):

1. Add the function to `ball_knower/io/loaders.py`
2. Follow the pattern of existing loaders (use `_resolve_file()`, normalize team names)
3. Add the new source to `load_all_sources()` if it should be loaded by default

---

## Compatibility Notes

### The src.data_loader Shim

The old `src.data_loader` module still exists as a **compatibility layer**. It now internally calls the unified loader, so old code continues to work:

```python
# This still works (but not recommended for new code)
from src import data_loader
df = data_loader.load_nfelo_power_ratings()
```

### Gradual Migration

You can migrate incrementally:
1. Start using unified loader in new scripts
2. Update old scripts when you touch them
3. Eventually deprecate `src.data_loader` once all code is migrated

---

## Common Migration Issues

### Issue 1: File Not Found

**Error:**
```
FileNotFoundError: Could not find file for category='power_ratings', provider='nfelo'...
```

**Solution:** Check that:
1. File exists in `data/current_season/`
2. File follows naming convention (either category-first or provider-first)
3. Season and week parameters match the actual filename

### Issue 2: Unmapped Team Names

**Warning:**
```
UserWarning: Found 3 rows with unmapped team names. These will be dropped.
```

**Solution:** Team name normalization is automatic, but if you see this warning:
1. Check `src/team_mapping.py` for the team name mapping
2. Add any missing team name variants to the mapping dict

### Issue 3: Multi-row Headers

Some CSV files (especially from Substack) have multi-row headers. The unified loader handles this automatically, but if you're using raw `pd.read_csv()`:

```python
# Manual handling
df = pd.read_csv(file, skiprows=1)  # Skip the first row
df = df.loc[:, ~df.columns.str.startswith('X.')]  # Remove junk columns
```

---

## Migration Checklist

When migrating a script:

- [ ] Replace `pd.read_csv()` calls with `loaders.load_*()` functions
- [ ] Replace `src.data_loader` imports with `ball_knower.io.loaders`
- [ ] Use `config.CURRENT_SEASON` and `config.CURRENT_WEEK` instead of hardcoded values
- [ ] Remove hardcoded file paths
- [ ] Test with both category-first and provider-first filenames (if applicable)
- [ ] Verify team names are normalized correctly
- [ ] Update any documentation or comments

---

## Questions?

If you run into issues during migration:

1. Check `ball_knower/io/loaders.py` for implementation details
2. Look at `run_demo.py` for working examples
3. Review this guide's code patterns
4. The unified loader provides helpful warnings and error messages

**Remember:** The unified loader is designed to make your life easier. If something seems harder than the old way, there's probably a better pattern available.

---

## Version History

- **v1.2** - Unified loader introduced, category-first naming recommended
- **v1.1** - Compatibility shim added to `src.data_loader`
- **v1.0** - Original ad-hoc loading system
