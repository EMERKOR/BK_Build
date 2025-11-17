# Ball Knower Data Migration Summary

## Current State (Phase 1 Complete)

### What Exists Now

**✅ New Loader System (Phase 1 - COMPLETE)**
- `ball_knower/io/loaders.py` - Unified data loading module
- `ball_knower/io/__init__.py` - Package interface
- `ball_knower/__init__.py` - Package root
- `docs/DATA_SOURCES.md` - Data source documentation
- `docs/DATA_MIGRATION_SUMMARY.md` - This migration summary

**📁 Current Data Files (Legacy Provider-First Pattern)**

All 10 CSV files in `data/current_season/` still use the old provider-first naming:

```
nfelo_power_ratings_2025_week_11.csv
nfelo_epa_tiers_off_def_2025_week_11.csv
nfelo_strength_of_schedule_2025_week_11.csv
nfelo_qb_rankings_2025_week_11.csv
nfelo_nfl_win_totals_2025_week_11 (1).csv
nfelo_nfl_receiving_leaders_2025_week_11.csv
substack_power_ratings_2025_week_11.csv
substack_qb_epa_2025_week_11.csv
substack_weekly_proj_elo_2025_week_11.csv
substack_weekly_proj_ppg_2025_week_11.csv
```

**🔧 Active Code (Still Using Legacy Loader)**

All 16 active Python files still import from `src/data_loader.py`:

| File | Status | Dependencies |
|------|--------|--------------|
| `run_demo.py` | Active | Uses `src.data_loader` |
| `ball_knower_v1_final.py` | Production | Uses `src.data_loader`, `src.features`, `src.models` |
| `ball_knower_v1_1.py` | Testing | Uses `src.data_loader`, `src.features`, `src.models` |
| `calibrate_*.py` (4 files) | Calibration | Uses `src.data_loader` |
| `test_*.py` (5 files) | Tests | Uses `src.data_loader` |
| `src/*.py` (7 files) | Core modules | Interdependent |

### Key Insight

**The new `ball_knower/io/loaders.py` module is fully implemented but NOT yet used by any production code.** It exists alongside the legacy system and can load data using either the new category-first OR legacy provider-first filename patterns via automatic fallback.

## Migration Plan

### ✅ Phase 1: Implement New Loader System (COMPLETE)

**Status:** Complete (Current)

**What Was Done:**
- ✅ Created `ball_knower/` package structure
- ✅ Implemented unified loaders with dual-pattern file resolution
- ✅ Added comprehensive documentation
- ✅ Loader works with current provider-first filenames via fallback

**No Breaking Changes:**
- ❌ Did NOT rename any CSV files
- ❌ Did NOT modify `src/data_loader.py`
- ❌ Did NOT change imports in existing scripts
- ❌ Did NOT touch `src/config.py` hardcoded paths

**Testing:**
The new loader system includes a built-in sanity check (`python -m ball_knower.io.loaders`) that validates:
- File resolution (tries category-first, falls back to provider-first)
- Data loading from all providers
- Team name normalization
- Merged ratings table generation

### 🔄 Phase 2: Add Compatibility Layer

**Status:** Not started

**Goal:** Make `src/data_loader.py` forward calls to new loaders with deprecation warnings.

**Tasks:**
1. Add backward-compatible wrapper functions to `src/data_loader.py`:
   ```python
   def load_nfelo_power_ratings(week, season, data_dir=None):
       warnings.warn(
           "load_nfelo_power_ratings() is deprecated. "
           "Use ball_knower.io.load_power_ratings('nfelo', week, season) instead.",
           DeprecationWarning
       )
       from ball_knower.io import load_power_ratings
       return load_power_ratings('nfelo', week, season, data_dir)
   ```

2. Update all existing loader functions similarly

3. Verify all existing scripts still work (with deprecation warnings)

**Estimated Time:** 2-3 hours

**Risk:** Low (adds warnings but maintains full backward compatibility)

### 🔄 Phase 3: Migrate Active Scripts

**Status:** Not started

**Goal:** Update all active scripts to use `ball_knower.io.loaders` directly.

**Priority Order:**
1. Test scripts (`test_*.py`) - Lowest risk, validates new system
2. Calibration scripts (`calibrate_*.py`) - Medium impact
3. Demo script (`run_demo.py`) - User-facing
4. Production models (`ball_knower_v1_final.py`, `ball_knower_v1_1.py`) - Highest stakes

**Migration Pattern:**
```python
# OLD
from src.data_loader import load_nfelo_power_ratings
ratings = load_nfelo_power_ratings(week=11, season=2025)

# NEW
from ball_knower.io import load_power_ratings
ratings = load_power_ratings('nfelo', week=11, season=2025)
```

**Testing Strategy:**
- Run full test suite after each script migration
- Compare outputs to ensure no behavior changes
- Run calibration and verify model coefficients unchanged

**Estimated Time:** 5-7 hours

**Risk:** Medium (requires thorough testing, but loaders are designed to be drop-in compatible)

### 📝 Phase 4: Rename CSV Files

**Status:** Not started

**Goal:** Rename all CSV files to category-first convention.

**⚠️ CRITICAL: Do NOT start Phase 4 until Phases 2-3 are complete and tested!**

**Renaming Map:**
```bash
# nfelo files
nfelo_power_ratings_2025_week_11.csv
  → power_ratings_nfelo_2025_week_11.csv

nfelo_epa_tiers_off_def_2025_week_11.csv
  → epa_tiers_nfelo_2025_week_11.csv

nfelo_strength_of_schedule_2025_week_11.csv
  → strength_of_schedule_nfelo_2025_week_11.csv

nfelo_qb_rankings_2025_week_11.csv
  → qb_rankings_nfelo_2025_week_11.csv

nfelo_nfl_win_totals_2025_week_11 (1).csv
  → win_totals_nfelo_2025_week_11.csv

nfelo_nfl_receiving_leaders_2025_week_11.csv
  → receiving_leaders_nfelo_2025_week_11.csv

# substack files
substack_power_ratings_2025_week_11.csv
  → power_ratings_substack_2025_week_11.csv

substack_qb_epa_2025_week_11.csv
  → qb_epa_substack_2025_week_11.csv

substack_weekly_proj_elo_2025_week_11.csv
  → weekly_projections_elo_substack_2025_week_11.csv

substack_weekly_proj_ppg_2025_week_11.csv
  → weekly_projections_ppg_substack_2025_week_11.csv
```

**After Renaming:**
- Remove fallback patterns from `ball_knower/io/loaders.py`
- Update `FALLBACK_FILENAMES` dict to be empty or remove it
- Verify all tests pass with new filenames

**Estimated Time:** 1-2 hours

**Risk:** Low (if Phases 2-3 complete, loaders already support new pattern)

### 🧹 Phase 5: Clean Up Legacy Code

**Status:** Not started

**Goal:** Remove or deprecate `src/data_loader.py` once migration is complete.

**Options:**

**Option A: Complete Removal**
- Delete `src/data_loader.py`
- Remove imports from all files
- Update `src/config.py` to reference new patterns
- Risk: High - ensures no old code remains

**Option B: Deprecation Stubs**
- Keep `src/data_loader.py` with minimal stubs that raise `DeprecationWarning`
- Prevents accidental imports but allows gradual transition
- Risk: Low - provides safety net

**Recommended:** Option B for at least one release cycle.

**Estimated Time:** 2-3 hours

**Risk:** Low (if Option B), Medium (if Option A)

## Timeline Summary

| Phase | Status | Estimated Time | Risk Level |
|-------|--------|---------------|------------|
| **Phase 1** | ✅ Complete | — | — |
| **Phase 2** | Pending | 2-3 hours | Low |
| **Phase 3** | Pending | 5-7 hours | Medium |
| **Phase 4** | Pending | 1-2 hours | Low |
| **Phase 5** | Pending | 2-3 hours | Low-Medium |
| **Total** | — | **11-16 hours** | — |

## Benefits After Migration

1. **Consistency:** All data files follow category-first naming convention
2. **Discoverability:** Easy to find all power ratings, all EPA files, etc.
3. **Scalability:** Easy to add new providers without filename collisions
4. **Unified API:** Single `ball_knower.io` interface for all data loading
5. **Type Safety:** Better IDE autocomplete and type hints
6. **Testing:** Isolated loader module easier to test independently

## Risk Mitigation

### Before Renaming Files (Phase 4)

**✅ Required Safeguards:**
- [ ] All test scripts migrated to new loaders (Phase 3)
- [ ] All production scripts migrated to new loaders (Phase 3)
- [ ] Full test suite passing with new loaders
- [ ] Calibration scripts verified to produce same outputs
- [ ] No references to old loader functions in active code

### During Migration

**✅ Best Practices:**
- Run tests after each script migration
- Keep git history clean with descriptive commits
- Tag releases before major changes (e.g., `pre-phase4-rename`)
- Document any behavior changes in commit messages

### Rollback Plan

If issues arise during migration:

1. **Phase 2-3:** Revert script changes, keep new loader available but unused
2. **Phase 4:** Rename files back to provider-first (simple `git mv` operations)
3. **Phase 5:** Restore `src/data_loader.py` from git history

## Current Dependencies Mapped

### File → Module → Data Dependencies

From the audit (see full dependency graph in audit report):

**Critical Path (Must Migrate First):**
```
run_demo.py
  → src.data_loader → All 10 CSV files
  → src.features
  → src.models

ball_knower_v1_final.py
  → src.data_loader → All 10 CSV files
  → src.features
  → src.models
```

**Supporting Path:**
```
calibrate_*.py (4 files)
  → src.data_loader → Various CSV files
  → src.features
  → src.models

test_*.py (5 files)
  → src.data_loader
  → Various src modules
```

### Hardcoded Paths in src/config.py

All 33 hardcoded file paths in `src/config.py` (lines 12-44) will need updating during Phase 4 when files are renamed. These paths are imported by `src/data_loader.py` and used throughout the codebase.

## Questions or Concerns?

- **"Will the new loaders work with my custom data?"**
  Yes! The loaders support custom `data_dir` paths and will work with any properly named CSV files.

- **"Can I use both old and new loaders during migration?"**
  Yes! Phases 2-3 are designed to allow gradual migration. The old loader will issue deprecation warnings.

- **"What if I find a bug in the new loaders?"**
  File an issue and we can add fixes before completing Phase 3. The fallback system makes this safe.

- **"Do I need to change how I call the loaders?"**
  Yes, but it's straightforward:
  ```python
  # Old: load_nfelo_power_ratings(week, season)
  # New: load_power_ratings('nfelo', week, season)
  ```

## Next Steps

**To proceed to Phase 2:**
1. Review this migration summary
2. Confirm Phase 1 loaders work correctly (run sanity check)
3. Begin implementing compatibility wrappers in `src/data_loader.py`
4. Test wrapper functions maintain backward compatibility

**Status as of:** Phase 1 complete, ready to begin Phase 2.
