# v1.2 Merge Validation Checklist

**Branch:** `claude/merge-validation-roadmap-01Sa7fRNybriSGy2XJiEAcry`
**Target:** `main` (baseline: `ef5ed02`)
**Date:** 2025-11-17

---

## Executive Summary

This checklist validates that the v1.2 branch can be safely merged into main without breaking references, imports, or existing functionality. The v1.2 branch introduces the unified loader architecture (`ball_knower.io.loaders`) and standardizes file naming conventions to category-first format.

### Changes Overview

**New Modules:**
- `ball_knower/__init__.py` - Package initialization
- `ball_knower/io/__init__.py` - IO subpackage initialization
- `ball_knower/io/loaders.py` - **Unified loader module (canonical API)**

**Data File Renames (Category-First Convention):**
- `nfelo_power_ratings_2025_week_11.csv` → `power_ratings_nfelo_2025_week_11.csv`
- `nfelo_epa_tiers_off_def_2025_week_11.csv` → `epa_tiers_nfelo_2025_week_11.csv`
- `nfelo_strength_of_schedule_2025_week_11.csv` → `strength_of_schedule_nfelo_2025_week_11.csv`
- `substack_power_ratings_2025_week_11.csv` → `power_ratings_substack_2025_week_11.csv`
- `substack_qb_epa_2025_week_11.csv` → `qb_epa_substack_2025_week_11.csv`
- `substack_weekly_proj_ppg_2025_week_11.csv` → `weekly_projections_ppg_substack_2025_week_11.csv`

**Modified Core Files:**
- `src/data_loader.py` - Updated with compatibility shim
- `src/config.py` - Configuration updates
- `test_data_loading.py` - Test updates
- `predict_current_week.py` - Active prediction script
- `rebuild_v1.py` - Active rebuild script
- `run_demo.py` - Active demo script

**Archive-Marked Scripts (Modified but Not Active):**
- `ball_knower_v1_1.py`
- `ball_knower_v1_final.py`
- `bk_v1_1_with_adjustments.py`
- `bk_v1_final.py`
- `calibrate_model.py`
- `calibrate_regression.py`
- `calibrate_simple.py`
- `calibrate_to_vegas.py`

---

## ✅ Validation Checklist

### 1. Import Compatibility

#### 1.1 Unified Loader Imports
- [ ] **Verify `ball_knower.io.loaders` is importable**
  ```bash
  python -c "from ball_knower.io import loaders; print('✓ Unified loader accessible')"
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

- [ ] **Verify all loader functions are accessible**
  ```bash
  python -c "from ball_knower.io import loaders; print(dir(loaders))"
  ```
  - Expected functions:
    - `load_power_ratings()`
    - `load_epa_tiers()`
    - `load_strength_of_schedule()`
    - `load_qb_epa()`
    - `load_weekly_projections_ppg()`
    - `load_all_sources()`
    - `merge_team_ratings()`
  - **Status:** PASS / FAIL
  - **Notes:**

#### 1.2 Legacy Compatibility
- [ ] **Verify `src.data_loader` still works**
  ```bash
  python -c "from src import data_loader; print('✓ Legacy loader accessible')"
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

- [ ] **Verify legacy loader forwards to unified loader when available**
  ```bash
  python -c "from src.data_loader import NEW_LOADERS_AVAILABLE; print(f'Unified loader available: {NEW_LOADERS_AVAILABLE}')"
  ```
  - **Expected:** `True`
  - **Status:** PASS / FAIL
  - **Notes:**

#### 1.3 Direct Module Imports
- [ ] **Verify team_mapping module**
  ```bash
  python -c "from src.team_mapping import normalize_team_name; print('✓ Team mapping works')"
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

- [ ] **Verify src module imports**
  ```bash
  python -c "from src import config, models, features; print('✓ Core modules accessible')"
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

---

### 2. Loader Functionality

#### 2.1 File Resolution (Category-First + Legacy Fallback)
- [ ] **Test category-first file resolution**
  ```bash
  python -c "
  from ball_knower.io.loaders import _resolve_file
  from pathlib import Path
  path = _resolve_file('power_ratings', 'nfelo', 2025, 11)
  expected = Path('data/current_season/power_ratings_nfelo_2025_week_11.csv')
  assert path.name == expected.name, f'Expected {expected.name}, got {path.name}'
  print(f'✓ Category-first resolution: {path.name}')
  "
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

- [ ] **Test legacy fallback (if legacy files exist)**
  - Temporarily rename a file to provider-first format
  - Verify loader finds it and issues deprecation warning
  - **Status:** PASS / FAIL / SKIPPED (no legacy files)
  - **Notes:**

#### 2.2 Data Loading
- [ ] **Load nfelo power ratings**
  ```bash
  python -c "
  from ball_knower.io import loaders
  df = loaders.load_power_ratings('nfelo', 2025, 11)
  print(f'✓ Loaded {len(df)} teams')
  assert 'team' in df.columns, 'Missing team column'
  assert 'nfelo' in df.columns, 'Missing nfelo column'
  "
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

- [ ] **Load all sources via orchestrator**
  ```bash
  python -c "
  from ball_knower.io import loaders
  data = loaders.load_all_sources(season=2025, week=11)
  expected_keys = ['power_ratings_nfelo', 'epa_tiers_nfelo', 'strength_of_schedule_nfelo',
                   'power_ratings_substack', 'qb_epa_substack', 'weekly_projections_ppg_substack',
                   'merged_ratings']
  for key in expected_keys:
      assert key in data, f'Missing key: {key}'
  print(f'✓ Loaded {len(data)} data sources')
  "
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

#### 2.3 Team Normalization
- [ ] **Verify team name normalization**
  ```bash
  python -c "
  from ball_knower.io import loaders
  df = loaders.load_power_ratings('nfelo', 2025, 11)
  # Check that team names match nfl_data_py format (e.g., 'KC', 'SF', not 'Kansas City')
  teams = df['team'].unique()
  assert all(len(t) <= 3 for t in teams if pd.notna(t)), 'Team names not normalized'
  print(f'✓ Teams normalized: {sorted(teams)[:5]}...')
  "
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

#### 2.4 Merged Ratings
- [ ] **Verify merged ratings structure**
  ```bash
  python -c "
  from ball_knower.io import loaders
  data = loaders.load_all_sources(2025, 11)
  merged = data['merged_ratings']
  print(f'✓ Merged {len(merged)} teams with {len(merged.columns)} columns')
  print(f'  Sample columns: {list(merged.columns)[:10]}')
  "
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

---

### 3. Active Script Compatibility

#### 3.1 run_demo.py
- [ ] **Execute run_demo.py**
  ```bash
  python run_demo.py
  ```
  - Should complete without import errors
  - Should use unified loader for team ratings
  - Should fall back to legacy loader for weekly projections
  - **Status:** PASS / FAIL
  - **Notes:**

#### 3.2 predict_current_week.py
- [ ] **Execute predict_current_week.py**
  ```bash
  python predict_current_week.py
  ```
  - Should complete without import errors
  - Uses `src.nflverse_data` directly (no loader dependency)
  - **Status:** PASS / FAIL
  - **Notes:**

#### 3.3 ball_knower_v1_2.py
- [ ] **Execute ball_knower_v1_2.py**
  ```bash
  python ball_knower_v1_2.py
  ```
  - Loads data directly from nfelo URL (no loader dependency)
  - Should complete without import errors
  - **Status:** PASS / FAIL
  - **Notes:**

#### 3.4 backtest_v1_2.py
- [ ] **Execute backtest_v1_2.py**
  ```bash
  python backtest_v1_2.py
  ```
  - Should complete without import errors
  - **Status:** PASS / FAIL
  - **Notes:**

#### 3.5 test_data_loading.py
- [ ] **Run data loading tests**
  ```bash
  python test_data_loading.py
  ```
  - Should test unified loader
  - Should verify all data sources load correctly
  - **Status:** PASS / FAIL
  - **Notes:**

---

### 4. Archive Script Status (Non-Breaking)

Archive-marked scripts are deprecated and won't be actively maintained after merge. They should still work due to legacy compatibility, but failures are non-blocking.

#### 4.1 Archive Scripts List
- [ ] **Confirm these scripts are marked as archived**
  - `ball_knower_v1_1.py` → Uses legacy `src.data_loader`
  - `ball_knower_v1_final.py` → Uses legacy `src.data_loader`
  - `bk_v1_1_with_adjustments.py` → Uses legacy `src.data_loader`
  - `bk_v1_final.py` → Uses legacy `src.data_loader`
  - `calibrate_model.py` → Uses legacy `src.data_loader`
  - `calibrate_regression.py` → Uses legacy `src.data_loader`
  - `calibrate_simple.py` → Uses legacy `src.data_loader`
  - `calibrate_to_vegas.py` → Uses legacy `src.data_loader`
  - **Status:** DOCUMENTED
  - **Notes:**

#### 4.2 Legacy Compatibility Shim
- [ ] **Verify archived scripts can still import**
  ```bash
  python -c "
  # Simulate archived script imports
  from src import data_loader, config, models
  print('✓ Archived scripts can still import legacy modules')
  "
  ```
  - **Status:** PASS / FAIL (non-blocking)
  - **Notes:**

---

### 5. Data File Integrity

#### 5.1 Category-First Files Exist
- [ ] **Verify all renamed files exist**
  ```bash
  ls -1 data/current_season/*.csv
  ```
  - Expected files:
    - ✓ `power_ratings_nfelo_2025_week_11.csv`
    - ✓ `epa_tiers_nfelo_2025_week_11.csv`
    - ✓ `strength_of_schedule_nfelo_2025_week_11.csv`
    - ✓ `power_ratings_substack_2025_week_11.csv`
    - ✓ `qb_epa_substack_2025_week_11.csv`
    - ✓ `weekly_projections_ppg_substack_2025_week_11.csv`
  - **Status:** PASS / FAIL
  - **Notes:**

#### 5.2 No Provider-First Files Remain
- [ ] **Verify old naming convention files are removed**
  ```bash
  ls data/current_season/nfelo_*.csv data/current_season/substack_*.csv 2>/dev/null || echo "✓ No legacy files found"
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

#### 5.3 Data Content Validation
- [ ] **Spot-check file contents haven't been corrupted**
  ```bash
  python -c "
  import pandas as pd
  df = pd.read_csv('data/current_season/power_ratings_nfelo_2025_week_11.csv')
  print(f'✓ Power ratings: {len(df)} teams, columns: {list(df.columns)[:5]}')
  "
  ```
  - **Status:** PASS / FAIL
  - **Notes:**

---

### 6. Branch-Merge Diff Summary

#### 6.1 Files Changed Count
- [ ] **Verify change scope**
  ```bash
  git diff ef5ed02..HEAD --stat
  ```
  - **Modified:** 13 files
  - **Added:** 3 files
  - **Renamed:** 6 files
  - **Total changes:** ~22 files
  - **Status:** DOCUMENTED
  - **Notes:**

#### 6.2 No Unexpected Changes
- [ ] **Review diff for unexpected modifications**
  ```bash
  git diff ef5ed02..HEAD --name-only | grep -v -E "ball_knower|data/current_season|src/|test_|run_demo|predict_|rebuild_|README"
  ```
  - Expected: No output (all changes are in expected locations)
  - **Status:** PASS / FAIL
  - **Notes:**

#### 6.3 No Breaking Deletions
- [ ] **Verify no critical files were deleted**
  ```bash
  git diff ef5ed02..HEAD --diff-filter=D --name-only
  ```
  - Expected: No deletions (files were renamed, not deleted)
  - **Status:** PASS / FAIL
  - **Notes:**

---

### 7. Cross-Branch Conflict Check

#### 7.1 Merge Conflict Simulation
- [ ] **Simulate merge with main**
  ```bash
  git checkout -b test-merge-validation
  git merge main --no-commit --no-ff 2>&1 | grep -E "CONFLICT|Automatic merge" || echo "✓ No conflicts detected"
  git merge --abort
  git checkout claude/merge-validation-roadmap-01Sa7fRNybriSGy2XJiEAcry
  git branch -D test-merge-validation
  ```
  - **Status:** PASS / FAIL / SKIPPED (no main branch)
  - **Notes:**

#### 7.2 Parallel Branch Check
- [ ] **Check for conflicts with `claude/review-model-features-*` branch**
  - That branch contains v1.3 and v1.4 models
  - No overlap expected (different model versions)
  - **Status:** NO CONFLICT EXPECTED
  - **Notes:**

---

### 8. Final Canonical Coverage Check

#### 8.1 All Feature Categories Covered
- [ ] **Verify loader supports all required categories**
  - ✓ `power_ratings` (nfelo, substack)
  - ✓ `epa_tiers` (nfelo)
  - ✓ `strength_of_schedule` (nfelo)
  - ✓ `qb_epa` (substack)
  - ✓ `weekly_projections_ppg` (substack)
  - **Status:** COMPLETE
  - **Notes:**

#### 8.2 Future Extension Readiness
- [ ] **Verify loader can handle new categories easily**
  - Category-first naming allows adding new categories (e.g., `market_lines_draftkings`)
  - File resolver pattern supports new providers
  - **Status:** READY
  - **Notes:**

---

## Validation Results Summary

### Critical Checks (Must Pass)
- [ ] All import compatibility checks pass
- [ ] Unified loader functions correctly
- [ ] Active scripts execute without errors
- [ ] Category-first data files exist and load correctly
- [ ] No breaking deletions or unexpected changes

### Non-Critical Checks (Should Pass)
- [ ] Archive scripts can still import (via compatibility shim)
- [ ] No merge conflicts detected
- [ ] Legacy fallback works (if legacy files exist)

---

## Sign-Off

**Validation Date:** _______________
**Validated By:** _______________
**Result:** PASS / FAIL / PASS WITH NOTES

**Notes:**

---

## Post-Merge Actions (if validation passes)

1. **Merge to main**
   ```bash
   git checkout main
   git merge claude/merge-validation-roadmap-01Sa7fRNybriSGy2XJiEAcry --no-ff
   git push origin main
   ```

2. **Tag release**
   ```bash
   git tag v1.2-unified-loader
   git push origin v1.2-unified-loader
   ```

3. **Update documentation**
   - Update README.md with unified loader usage examples
   - Mark archive scripts as deprecated in documentation

4. **Communicate changes**
   - Notify team of new canonical loader API
   - Deprecation warning for `src.data_loader` (still works, but use `ball_knower.io.loaders`)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
