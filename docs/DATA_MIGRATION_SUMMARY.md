# Data Migration Summary

## Overview

This document summarizes the migration from the legacy provider-first data loading system to the new unified category-first loader architecture.

**Migration Date:** 2025-11-17
**Branch:** `claude/review-instructions-01CqnEpFcFDjNtucvgvX5tZm`
**Status:** Phases 1 & 2 Complete

---

## Migration Phases

### ✅ Phase 1: Create Unified Loaders (COMPLETE)

**Goal:** Build `ball_knower.io.loaders` as the new unified data loading interface.

**Deliverables:**
- ✅ `ball_knower/` package structure
- ✅ `ball_knower/io/loaders.py` (562 lines)
- ✅ Category-specific loader functions
- ✅ Dual-pattern file resolution (category-first with provider-first fallback)
- ✅ `load_all_sources()` orchestrator
- ✅ `merge_team_ratings()` function
- ✅ Team name normalization integration
- ✅ Deprecation warnings for legacy filenames
- ✅ Documentation (`docs/DATA_SOURCES.md`)

**Duration:** ~3 hours

**Files Created:**
```
ball_knower/
├── __init__.py                 (7 lines)
└── io/
    ├── __init__.py             (8 lines)
    └── loaders.py              (562 lines)

docs/
└── DATA_SOURCES.md             (295 lines)
```

**Key Functions:**
- `load_power_ratings(provider, season, week, data_dir)`
- `load_epa_tiers(provider, season, week, data_dir)`
- `load_strength_of_schedule(provider, season, week, data_dir)`
- `load_qb_rankings(provider, season, week, data_dir)`
- `load_qb_epa(provider, season, week, data_dir)`
- `load_weekly_projections_ppg(provider, season, week, data_dir)`
- `load_weekly_projections_elo(provider, season, week, data_dir)`
- `load_win_totals(provider, season, week, data_dir)`
- `load_receiving_leaders(provider, season, week, data_dir)`
- `load_all_sources(week, season, data_dir)` → Returns dict with exact keys for Phase 2
- `merge_team_ratings(data_dict)` → Returns merged DataFrame

---

### ✅ Phase 2: Create Compatibility Layer (COMPLETE)

**Goal:** Transform `src/data_loader.py` to forward calls to new loaders while maintaining backward compatibility.

**Deliverables:**
- ✅ `NEW_LOADERS_AVAILABLE` flag with graceful import fallback
- ✅ All existing functions renamed to `_legacy_*` helpers
- ✅ Public wrapper functions with deprecation warnings
- ✅ Orchestrator functions updated (`load_all_current_week_data`, `merge_current_week_ratings`)
- ✅ Key mapping for backward compatibility
- ✅ Zero breaking changes to existing code
- ✅ Documentation (`docs/PHASE2_SUMMARY.md`)

**Duration:** ~2 hours

**Files Modified:**
```
src/
└── data_loader.py              (+280 lines, refactored)

docs/
├── PHASE2_SUMMARY.md           (317 lines)
└── PHASE1_NOTE.md              (60 lines)
```

**Pattern Example:**
```python
def load_nfelo_power_ratings():
    """DEPRECATED: Use ball_knower.io.loaders.load_power_ratings('nfelo', ...) instead."""
    warnings.warn(..., DeprecationWarning)

    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_power_ratings("nfelo", CURRENT_SEASON, CURRENT_WEEK)

    return _legacy_load_nfelo_power_ratings()
```

---

### ⏳ Phase 3: Update Script Imports (PENDING)

**Goal:** Migrate all scripts to use `ball_knower.io.loaders` directly instead of `src.data_loader`.

**Scripts to Update:**
- `ball_knower_v1_final.py`
- `bk_v1_final.py`
- `ball_knower_v1_1.py`
- `run_demo.py`
- `test_data_loading.py`
- `calibrate_*.py` files
- Any other scripts using `src.data_loader`

**Changes:**
```python
# OLD (deprecated)
from src import data_loader
data = data_loader.load_nfelo_power_ratings()

# NEW (recommended)
from ball_knower.io import loaders
data = loaders.load_power_ratings("nfelo", season=2025, week=11)
```

**Estimated Duration:** 1-2 hours

---

### ⏳ Phase 4: Rename CSV Files (PENDING)

**Goal:** Rename all CSV files from provider-first to category-first naming.

**File Renaming Map:**

| Current (Provider-First) | Target (Category-First) |
|-------------------------|------------------------|
| `nfelo_power_ratings_2025_week_11.csv` | `power_ratings_nfelo_2025_week_11.csv` |
| `nfelo_epa_tiers_off_def_2025_week_11.csv` | `epa_tiers_nfelo_2025_week_11.csv` |
| `nfelo_strength_of_schedule_2025_week_11.csv` | `strength_of_schedule_nfelo_2025_week_11.csv` |
| `nfelo_qb_rankings_2025_week_11.csv` | `qb_rankings_nfelo_2025_week_11.csv` |
| `nfelo_nfl_win_totals_2025_week_11 (1).csv` | `win_totals_nfelo_2025_week_11.csv` |
| `nfelo_nfl_receiving_leaders_2025_week_11.csv` | `receiving_leaders_nfelo_2025_week_11.csv` |
| `substack_power_ratings_2025_week_11.csv` | `power_ratings_substack_2025_week_11.csv` |
| `substack_qb_epa_2025_week_11.csv` | `qb_epa_substack_2025_week_11.csv` |
| `substack_weekly_proj_ppg_2025_week_11.csv` | `weekly_projections_ppg_substack_2025_week_11.csv` |
| `substack_weekly_proj_elo_2025_week_11.csv` | `weekly_projections_elo_substack_2025_week_11.csv` |

**Commands:**
```bash
cd data/current_season
mv nfelo_power_ratings_2025_week_11.csv power_ratings_nfelo_2025_week_11.csv
mv nfelo_epa_tiers_off_def_2025_week_11.csv epa_tiers_nfelo_2025_week_11.csv
# ... (continue for all files)
```

**Estimated Duration:** 30 minutes

**Note:** Once renamed, deprecation warnings will stop appearing during data loading.

---

### ⏳ Phase 5: Clean Up Legacy Code (PENDING)

**Goal:** Remove legacy implementations and update `src/config.py`.

**Tasks:**
1. Remove all `_legacy_*` functions from `src/data_loader.py`
2. Remove compatibility wrappers (keep only historical loaders)
3. Update `src/config.py` to remove hardcoded file paths (use category-first patterns)
4. Update any remaining documentation references

**Estimated Duration:** 1-2 hours

---

## Current State (Post Phase 1 & 2)

### File Organization

**Old Naming (Current CSV Files):**
```
data/current_season/
├── nfelo_power_ratings_2025_week_11.csv         ← provider-first
├── nfelo_epa_tiers_off_def_2025_week_11.csv     ← provider-first
├── substack_power_ratings_2025_week_11.csv      ← provider-first
└── ...
```

**New Naming (Target):**
```
data/current_season/
├── power_ratings_nfelo_2025_week_11.csv         ← category-first
├── epa_tiers_nfelo_2025_week_11.csv             ← category-first
├── power_ratings_substack_2025_week_11.csv      ← category-first
└── ...
```

### Loader Behavior

**Current Behavior (Phases 1 & 2 Complete):**

1. **New unified loaders** (`ball_knower.io.loaders`):
   - Try category-first filename first
   - Fall back to provider-first if needed
   - Issue deprecation warning when using legacy files
   - Return normalized DataFrames

2. **Compatibility layer** (`src/data_loader`):
   - Check if new loaders available (`NEW_LOADERS_AVAILABLE`)
   - Forward to new loaders if available
   - Emit deprecation warning for legacy API
   - Fall back to `_legacy_*` functions if new loaders unavailable
   - Maintain exact same return values/types

3. **Existing scripts**:
   - Continue to work unchanged
   - See two types of warnings:
     - API deprecation (use new loaders)
     - File naming deprecation (rename CSV files)

### Key Mappings

**load_all_sources() Return Keys:**
```python
{
    'power_ratings_nfelo': DataFrame,
    'epa_tiers_nfelo': DataFrame,
    'strength_of_schedule_nfelo': DataFrame,
    'power_ratings_substack': DataFrame,
    'qb_epa_substack': DataFrame,
    'weekly_projections_ppg_substack': DataFrame,
    'merged_ratings': DataFrame,
}
```

**Legacy load_all_current_week_data() Keys (mapped from new loaders):**
```python
{
    'nfelo_power': power_ratings_nfelo,         # ← key mapping
    'nfelo_epa': epa_tiers_nfelo,               # ← key mapping
    'nfelo_sos': strength_of_schedule_nfelo,    # ← key mapping
    'substack_power': power_ratings_substack,   # ← key mapping
    'substack_qb_epa': qb_epa_substack,         # ← key mapping
    'substack_weekly': weekly_projections_ppg_substack,  # ← key mapping
    'coaches': load_head_coaches(),             # ← still uses legacy
}
```

---

## Testing Status

### ✅ Tests Passing

**test_data_loading.py:**
```bash
✓ Config test passed
✓ Team normalization test passed
✓ Data loading test passed
✓✓✓ ALL TESTS PASSED ✓✓✓
```

**run_demo.py:**
```bash
✓ Loaded Week 11 data
✓ Merged team ratings (32 teams)
✓ Generated predictions
✓ Found 13 value bets
```

**ball_knower.io.loaders sanity check:**
```bash
✓ power_ratings_nfelo: 32 rows, 22 columns
✓ epa_tiers_nfelo: 32 rows, 6 columns
✓ strength_of_schedule_nfelo: 32 rows, 13 columns
✓ power_ratings_substack: 32 rows, 11 columns
✓ qb_epa_substack: 61 rows, 10 columns
✓ weekly_projections_ppg_substack: 15 rows, 10 columns
✓ merged_ratings: 32 rows, 10 columns
```

### Warnings Observed

**API Deprecation Warnings (Expected):**
```
DeprecationWarning: load_nfelo_power_ratings() is deprecated and will be removed
in a future version. Use ball_knower.io.loaders.load_power_ratings('nfelo', ...) instead.
```

**File Naming Deprecation Warnings (Expected):**
```
DeprecationWarning: Using legacy filename 'nfelo_power_ratings_2025_week_11.csv'.
Consider renaming to 'power_ratings_nfelo_2025_week_11.csv' for consistency.
```

---

## Risk Assessment

### Low Risk
- ✅ Phase 1: New loaders implemented, tested, working
- ✅ Phase 2: Compatibility layer working, zero breaking changes
- ✅ Graceful fallbacks ensure backward compatibility

### Medium Risk
- ⚠️ Phase 3: Script updates could introduce import errors if done incorrectly
- **Mitigation:** Update one script at a time, test after each change

### Low Risk
- ✅ Phase 4: File renaming is reversible (just rename back)
- ✅ Phase 5: Legacy code can be kept in git history

---

## Rollback Plan

### If Issues Arise After Phase 1 & 2

```bash
# Revert to pre-migration state
git checkout HEAD~1 -- src/data_loader.py
git checkout HEAD~1 -- ball_knower/
git checkout HEAD~1 -- docs/
```

### If Issues Arise After Phase 3

```bash
# Revert individual scripts
git checkout HEAD~1 -- run_demo.py
# Continue using src.data_loader (compatibility layer still works)
```

### If Issues Arise After Phase 4

```bash
# Rename files back to provider-first
cd data/current_season
mv power_ratings_nfelo_2025_week_11.csv nfelo_power_ratings_2025_week_11.csv
# ... (fallback still works in loaders)
```

---

## Timeline Estimate

| Phase | Description | Duration | Status |
|-------|-------------|----------|--------|
| 1 | Unified loaders | 3 hours | ✅ Complete |
| 2 | Compatibility layer | 2 hours | ✅ Complete |
| 3 | Update script imports | 1-2 hours | ⏳ Pending |
| 4 | Rename CSV files | 30 min | ⏳ Pending |
| 5 | Clean up legacy code | 1-2 hours | ⏳ Pending |
| **Total** | **End-to-end migration** | **8-10 hours** | **40% Complete** |

---

## Benefits

### Immediate (Phases 1 & 2)
- ✅ Unified, consistent API for data loading
- ✅ Automatic team name normalization
- ✅ Graceful error handling with fallback
- ✅ Clear deprecation path for legacy code
- ✅ Comprehensive documentation

### After Phase 3
- ✅ Cleaner, more maintainable script code
- ✅ Easier to add new data sources
- ✅ Consistent function signatures across codebase

### After Phase 4
- ✅ Category-first naming makes data easier to find
- ✅ No more deprecation warnings
- ✅ Consistent file organization

### After Phase 5
- ✅ Simplified `src/data_loader.py`
- ✅ Reduced code duplication
- ✅ Cleaner codebase for future development

---

## Next Steps

1. **Complete Phase 3:** Update all scripts to use `ball_knower.io.loaders`
2. **Complete Phase 4:** Rename CSV files to category-first naming
3. **Complete Phase 5:** Remove legacy code from `src/data_loader.py`
4. **Update documentation:** Mark migration as complete
5. **Celebrate!** 🎉

---

**Last Updated:** 2025-11-17
**Branch:** `claude/review-instructions-01CqnEpFcFDjNtucvgvX5tZm`
**Phases Complete:** 2 of 5 (40%)
