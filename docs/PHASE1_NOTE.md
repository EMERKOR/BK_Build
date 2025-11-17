# Phase 1 Note

## Status: Not Yet on This Branch

Phase 1 (creating `ball_knower.io.loaders`) has not been implemented on branch `claude/review-instructions-01CqnEpFcFDjNtucvgvX5tZm`.

## What Phase 1 Should Create

Based on the migration plan, Phase 1 should create:

1. **`ball_knower/` package structure:**
   ```
   ball_knower/
   ├── __init__.py
   └── io/
       ├── __init__.py
       └── loaders.py  # ~632 lines
   ```

2. **`ball_knower/io/loaders.py` should contain:**
   - Dual-pattern file resolution (`_resolve_file_path()`)
   - Team name normalization (`_normalize_team_column_inplace()`)
   - 9 category-specific loaders:
     - `load_power_ratings(provider, season, week)`
     - `load_epa_tiers(provider, season, week)`
     - `load_strength_of_schedule(provider, season, week)`
     - `load_qb_rankings(provider, season, week)`
     - `load_qb_epa(provider, season, week)`
     - `load_weekly_projections_ppg(provider, season, week)`
     - `load_weekly_projections_elo(provider, season, week)`
     - `load_win_totals(provider, season, week)`
     - `load_receiving_leaders(provider, season, week)`
   - Orchestrator functions:
     - `load_all_sources(week, season)` → returns dict
     - `merge_team_ratings(data_dict)` → returns DataFrame
   - Fallback filename patterns for current dataset

3. **Documentation:**
   - `docs/DATA_SOURCES.md` (204 lines)
   - `docs/DATA_MIGRATION_SUMMARY.md` (317 lines)

## Current State

- **Phase 2 is complete** on this branch
- Phase 2 gracefully handles Phase 1's absence via `NEW_LOADERS_AVAILABLE` flag
- All code falls back to legacy implementations when `ball_knower.io.loaders` doesn't exist
- **No breaking changes** - everything works as before

## Integration

Once Phase 1 is created or merged:

1. `NEW_LOADERS_AVAILABLE` will automatically become `True`
2. All `src.data_loader` functions will forward to `ball_knower.io.loaders`
3. No code changes needed in Phase 2
4. Deprecation warnings will guide users to new API

## Option to Create Phase 1 on This Branch

If Phase 1 needs to be created on this branch, follow the implementation described in the migration plan and create the `ball_knower/` package structure with the unified loaders.

---

**Note:** This document is for reference only. Phase 2 (compatibility layer) is complete and ready for testing once Phase 1 is available.
