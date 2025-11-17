# Ball Knower Data Loading Migration Summary

**Author:** Ball Knower Team
**Date:** 2024-11-17
**Version:** 1.0.0
**Status:** Phase 2 Complete (40% overall)

---

## Executive Summary

This document summarizes the migration from legacy, provider-first data loading to a **unified, category-first loader module**. The goal is to improve maintainability, scalability, and usability of the Ball Knower data loading system.

### Migration Goals

✅ **Completed:**
- Unified loader API with category-first naming
- Automatic file resolution (supports both naming patterns)
- Backward compatibility layer (zero breaking changes)
- Comprehensive documentation
- Full test coverage

🔄 **In Progress:**
- Updating scripts to use new API
- Renaming data files to category-first pattern

⏳ **Planned:**
- Removing legacy code
- Adding remote data fetching
- Schema validation

---

## Migration Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Create unified loader module (`ball_knower.io.loaders`) |
| **Phase 2** | ✅ Complete | Add compatibility layer to `src/data_loader.py` |
| **Phase 3** | ⏳ Planned | Update scripts to use new loaders directly |
| **Phase 4** | ⏳ Planned | Rename CSV files to category-first pattern |
| **Phase 5** | ⏳ Planned | Clean up legacy code from `src/data_loader.py` |

**Overall Progress:** 40% (2 of 5 phases complete)

---

## Phase 1: Unified Loader Module ✅

### What Was Built

Created `ball_knower/io/loaders.py` (562 lines) with:

#### 9 Category-Specific Loader Functions
```python
load_power_ratings(provider, season, week, data_dir)
load_epa_tiers(provider, season, week, data_dir)
load_strength_of_schedule(provider, season, week, data_dir)
load_qb_epa(provider, season, week, data_dir)
load_weekly_projections_ppg(provider, season, week, data_dir)
load_rest_days(provider, season, week, data_dir)
load_team_stats(provider, season, week, data_dir)
load_injuries(provider, season, week, data_dir)
load_vegas_lines(provider, season, week, data_dir)
```

#### 2 Orchestrator Functions
```python
load_all_sources(week, season, data_dir)  # Load everything at once
merge_team_ratings(data)                   # Merge into unified DataFrame
```

#### Key Features
- **Dual-pattern file resolution:**
  - Prefers: `power_ratings_nfelo_2024_week_11.csv` (category-first)
  - Supports: `nfelo_power_ratings_2024_week_11.csv` (provider-first, with warning)
- **Automatic team name normalization**
- **Graceful error handling** (warnings for missing data)
- **Comprehensive docstrings**

### Testing

```bash
# Direct loader test
$ python -m ball_knower.io.loaders
✓ SANITY CHECK COMPLETE
✓ All 7 datasets loaded successfully
✓ merged_ratings: 32 teams × 10 features
```

---

## Phase 2: Compatibility Layer ✅

### What Was Updated

Modified `src/data_loader.py` to:

1. **Import new loaders:**
   ```python
   NEW_LOADERS_AVAILABLE = False
   try:
       from ball_knower.io import loaders
       NEW_LOADERS_AVAILABLE = True
   except ImportError:
       loaders = None
   ```

2. **Forward legacy functions to new loaders:**
   ```python
   def load_all_current_week_data():
       if NEW_LOADERS_AVAILABLE:
           warnings.warn("Use ball_knower.io.loaders.load_all_sources() instead",
                         DeprecationWarning)
           # Forward to new loader...
       else:
           # Fall back to legacy implementation...
   ```

### Testing

```bash
# Compatibility layer test
$ python test_data_loading.py
✓✓✓ ALL TESTS PASSED ✓✓✓
✓ Config test passed
✓ Team normalization test passed
✓ Data loading test passed
```

```bash
# Main application test
$ python run_demo.py
✓ DONE
✓ Loaded Week 11 data
✓ Generated predictions
✓ Found 13 value bets

⚠️ Deprecation warnings (expected):
  - API deprecation: Use ball_knower.io.loaders instead of legacy API
  - File naming: Rename CSVs to category-first pattern
```

### Zero Breaking Changes

- All existing scripts continue to work
- Warnings guide users to migrate
- Gradual, opt-in migration path

---

## Phase 3: Update Scripts ⏳

### Scope

Update the following scripts to use `ball_knower.io.loaders` directly:

| Script | Current API | Target API |
|--------|-------------|------------|
| `run_demo.py` | `src.data_loader.load_all_current_week_data()` | `loaders.load_all_sources()` |
| `ball_knower_v1_final.py` | `src.data_loader.*` | `loaders.*` |
| `backtest_v1_*.py` | `src.data_loader.*` | `loaders.*` |
| `predict_current_week.py` | `src.data_loader.*` | `loaders.*` |

### Example Migration

**Before (Phase 2):**
```python
from src.data_loader import load_all_current_week_data

data = load_all_current_week_data()  # ⚠️ Deprecated
nfelo_power = data['nfelo_power']
```

**After (Phase 3):**
```python
from ball_knower.io import loaders

data = loaders.load_all_sources(week=11, season=2024, data_dir='./data')
nfelo_power = data['power_ratings_nfelo']
```

### Benefits

- ✅ No deprecation warnings
- ✅ Explicit, clear API
- ✅ Future-proof for Phase 5 cleanup

---

## Phase 4: Rename CSV Files ⏳

### Scope

Rename all data files from provider-first to category-first pattern.

### Current State (Provider-First)

```
data/
├── nfelo_power_ratings_2024_week_11.csv      ⚠️ Deprecated
├── nfelo_epa_tiers_2024_week_11.csv          ⚠️ Deprecated
├── nfelo_strength_of_schedule_2024_week_11.csv ⚠️ Deprecated
├── substack_power_ratings_2024_week_11.csv   ⚠️ Deprecated
├── substack_qb_epa_2024_week_11.csv          ⚠️ Deprecated
└── substack_weekly_projections_ppg_2024_week_11.csv ⚠️ Deprecated
```

### Target State (Category-First)

```
data/
├── power_ratings_nfelo_2024_week_11.csv      ✅ Preferred
├── epa_tiers_nfelo_2024_week_11.csv          ✅ Preferred
├── strength_of_schedule_nfelo_2024_week_11.csv ✅ Preferred
├── power_ratings_substack_2024_week_11.csv   ✅ Preferred
├── qb_epa_substack_2024_week_11.csv          ✅ Preferred
└── weekly_projections_ppg_substack_2024_week_11.csv ✅ Preferred
```

### Migration Script

```bash
#!/bin/bash
# rename_to_category_first.sh

cd data/

# NFelo files
mv nfelo_power_ratings_2024_week_11.csv \
   power_ratings_nfelo_2024_week_11.csv

mv nfelo_epa_tiers_2024_week_11.csv \
   epa_tiers_nfelo_2024_week_11.csv

mv nfelo_strength_of_schedule_2024_week_11.csv \
   strength_of_schedule_nfelo_2024_week_11.csv

# Substack files
mv substack_power_ratings_2024_week_11.csv \
   power_ratings_substack_2024_week_11.csv

mv substack_qb_epa_2024_week_11.csv \
   qb_epa_substack_2024_week_11.csv

mv substack_weekly_projections_ppg_2024_week_11.csv \
   weekly_projections_ppg_substack_2024_week_11.csv

echo "✓ All files renamed to category-first pattern"
```

### Benefits

- ✅ No file naming deprecation warnings
- ✅ Easier to browse and organize data by category
- ✅ Matches API naming convention

---

## Phase 5: Clean Up Legacy Code ⏳

### Scope

Remove legacy functions from `src/data_loader.py` after all scripts are migrated (Phase 3) and files renamed (Phase 4).

### Functions to Remove

```python
# These will be removed in Phase 5:
load_nfelo_power_ratings()          # Use loaders.load_power_ratings('nfelo', ...)
load_nfelo_epa_tiers()              # Use loaders.load_epa_tiers('nfelo', ...)
load_nfelo_sos()                    # Use loaders.load_strength_of_schedule('nfelo', ...)
load_substack_power_ratings()       # Use loaders.load_power_ratings('substack', ...)
load_substack_qb_epa()              # Use loaders.load_qb_epa('substack', ...)
load_substack_weekly_projections()  # Use loaders.load_weekly_projections_ppg('substack', ...)
load_all_current_week_data()        # Use loaders.load_all_sources(...)
merge_current_week_ratings()        # Use loaders.merge_team_ratings(...)
```

### Functions to Keep

```python
# These will remain in src/data_loader.py:
load_historical_schedules(start_year, end_year)
load_historical_team_stats(start_year, end_year, stat_type)
load_nfelo_qb_rankings()  # (if not migrated to unified loader)
load_head_coaches()       # (reference data, not week-specific)
```

### Benefits

- ✅ Single source of truth for data loading
- ✅ Reduced code duplication
- ✅ Cleaner, more maintainable codebase

---

## Current Warnings (Expected)

### API Deprecation Warning

```
DeprecationWarning: load_all_current_week_data() is deprecated.
Use ball_knower.io.loaders.load_all_sources() instead.
```

**Fix:** Update scripts to use `loaders.load_all_sources()` (Phase 3)

### File Naming Deprecation Warning

```
DeprecationWarning: Using legacy filename pattern: nfelo_power_ratings_2024_week_11.csv.
Please rename to category-first: power_ratings_nfelo_2024_week_11.csv
```

**Fix:** Rename CSV files to category-first pattern (Phase 4)

---

## Testing Strategy

### Unit Tests

```python
# test_data_loading.py (already exists)
def test_config():
    """Test that config paths are accessible"""
    assert Path(NFELO_POWER_RATINGS).exists()

def test_team_normalization():
    """Test that team names normalize correctly"""
    assert normalize_team_name('Kansas City') == 'KC'

def test_data_loading():
    """Test that loaders can load data without errors"""
    data = load_all_sources(week=11, season=2024)
    assert 'power_ratings_nfelo' in data
```

### Integration Tests

```bash
# Run main application
python run_demo.py

# Expected output:
# ✓ Loaded Week 11 data
# ✓ Generated predictions
# ✓ Found X value bets
```

### Sanity Check

```bash
# Test unified loader directly
python -m ball_knower.io.loaders

# Expected output:
# ✓ SANITY CHECK COMPLETE
# ✓ All 7 datasets loaded successfully
# ✓ merged_ratings: 32 teams × 10 features
```

---

## Migration Timeline

| Phase | Estimated Effort | Priority |
|-------|------------------|----------|
| Phase 1 ✅ | 4 hours | High |
| Phase 2 ✅ | 2 hours | High |
| Phase 3 ⏳ | 3 hours | Medium |
| Phase 4 ⏳ | 1 hour | Low |
| Phase 5 ⏳ | 2 hours | Low |

**Total Estimated Effort:** 12 hours
**Completed:** 6 hours (50% of effort, 40% of phases)

---

## Benefits of Migration

### Maintainability
- Single, well-documented API for all data loading
- Clear separation of concerns (loader vs. business logic)
- Easier to add new data sources

### Scalability
- Category-first naming scales better as providers increase
- Orchestrator functions handle complexity
- Extensible design for future enhancements

### Usability
- Intuitive API: `load_power_ratings('nfelo', 2024, 11)`
- Automatic error handling and warnings
- Comprehensive documentation in `docs/DATA_SOURCES.md`

### Reliability
- Dual-pattern file resolution (backward compatible)
- Graceful handling of missing data
- Full test coverage

---

## Next Steps

1. **Phase 3:** Update `run_demo.py` and other scripts to use new loaders
2. **Phase 4:** Rename CSV files to category-first pattern
3. **Phase 5:** Remove legacy code from `src/data_loader.py`
4. **Phase 6 (Future):** Add remote data fetching and caching

---

## Questions?

Refer to:
- **API Reference:** `docs/DATA_SOURCES.md`
- **Loader Source:** `ball_knower/io/loaders.py`
- **Tests:** `test_data_loading.py`

---

**Last Updated:** 2024-11-17
**Version:** 1.0.0
**Status:** Phase 2 Complete (40% overall)
