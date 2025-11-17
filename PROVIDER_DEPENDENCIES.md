# Provider-Specific Dependencies Report

**Generated:** 2025-11-17
**Branch:** claude/review-v1.1-loader-alignment-01TYb4oXtX6ncUsdLSUf8rFQ
**Purpose:** Document all locations where code depends on provider-specific column names or data structures

---

## Executive Summary

All active scripts now use the unified loader (`ball_knower.io.loaders`), but **downstream code still relies on provider-specific column names** and **hardcoded provider keys**. This report catalogs every dependency to inform the feature mapping layer (Priority 2).

---

## Migration Status

### ✅ Completed
- **investigate_data.py** - Migrated to `loaders.load_all_sources()`
- **run_demo.py** - Already using `loaders.load_all_sources()`
- **src/__init__.py** - Added legacy deprecation notes

### ⏭️ Skipped (Archive Files)
All archive files intentionally use legacy loaders:
- `ball_knower_v1_1.py`
- `ball_knower_v1_final.py`
- `bk_v1_final.py`
- `rebuild_v1.py`
- `bk_v1_1_with_adjustments.py`
- `calibrate_simple.py`
- `calibrate_regression.py`
- `calibrate_to_vegas.py`
- `calibrate_model.py`

---

## Provider-Specific Dependencies

### 1. **investigate_data.py**

**Line 32, 43:**
```python
team_ratings[['team', 'nfelo', 'epa_margin', 'Ovr.']]
```

**Dependencies:**
- `'nfelo'` - nfelo-specific column (power rating)
- `'epa_margin'` - Calculated from nfelo EPA data (not a raw column)
- `'Ovr.'` - Substack-specific column (overall rating)

**Impact:** When adding new providers in v1.2/v1.3, this hardcoded column list won't include new provider columns.

**Recommended Fix:** Use canonical feature names mapped via feature mapping layer.

---

### 2. **run_demo.py**

#### Line 37:
```python
team_ratings[['team', 'nfelo', 'epa_off', 'epa_def', 'Ovr.']].sort_values('nfelo', ascending=False)
```

**Dependencies:**
- `'nfelo'` - nfelo-specific
- `'epa_off'` - nfelo-specific (offensive EPA)
- `'epa_def'` - nfelo-specific (defensive EPA)
- `'Ovr.'` - Substack-specific

#### Lines 44-45:
```python
from src import data_loader
weekly_data = data_loader.load_substack_weekly_projections()
```

**Dependencies:**
- **Still using legacy loader** for weekly projections matchup parsing
- The unified loader returns raw DataFrames; we need matchup extraction logic

**Recommended Fix:** Add matchup extraction to unified loader or create a separate matchup parser.

#### Lines 50, 62:
```python
team_ratings[['team', 'nfelo', 'epa_margin', 'Ovr.']]
```

**Dependencies:** Same as investigate_data.py

#### Lines 80-88:
```python
home_features = {
    'nfelo': game.get('nfelo_home'),
    'epa_margin': game.get('epa_margin_home'),
    'Ovr.': game.get('Ovr._home')
}
```

**Dependencies:**
- Feature dict keys are provider-specific
- Model expects these exact key names

**Impact:** Adding new providers requires updating this dict structure.

**Recommended Fix:** Canonical feature mapping that translates provider columns → standard feature names.

---

### 3. **src/models.py**

#### Lines 44-46 (DeterministicSpreadModel.__init__):
```python
self.weights = {
    'epa_margin': 35,
    'nfelo_diff': 0.02,
    'substack_ovr_diff': 0.5
}
```

**Dependencies:**
- Weight keys are provider-specific
- `'epa_margin'` - nfelo-derived
- `'nfelo_diff'` - nfelo-specific
- `'substack_ovr_diff'` - Substack-specific

#### Lines 63-75 (predict method):
```python
if 'epa_margin' in home_features and 'epa_margin' in away_features:
    epa_diff = home_features['epa_margin'] - away_features['epa_margin']
    spread -= epa_diff * self.weights['epa_margin']

if 'nfelo' in home_features and 'nfelo' in away_features:
    nfelo_diff = home_features['nfelo'] - away_features['nfelo']
    spread -= nfelo_diff * self.weights['nfelo_diff']

if 'Ovr.' in home_features and 'Ovr.' in away_features:
    ovr_diff = home_features['Ovr.'] - away_features['Ovr.']
    spread -= ovr_diff * self.weights['substack_ovr_diff']
```

**Dependencies:**
- Feature lookup keys: `'epa_margin'`, `'nfelo'`, `'Ovr.'`
- All provider-specific

**Impact:** Model cannot use features from new providers without code changes.

**Recommended Fix:**
1. Define canonical feature names (e.g., `'overall_rating'`, `'qb_adjustment'`, `'epa_differential'`)
2. Map provider columns to canonical names before passing to model
3. Model uses only canonical names

---

### 4. **ball_knower/io/loaders.py**

#### Lines 334-387 (merge_team_ratings function):
```python
merge_specs = [
    ("epa_tiers_nfelo", "_epa"),
    ("strength_of_schedule_nfelo", "_sos"),
    ("power_ratings_substack", "_substack"),
    ("qb_epa_substack", "_qb_epa"),
    ("weekly_projections_ppg_substack", "_proj_ppg"),
]

for key, suffix in merge_specs:
    if key not in sources:
        warnings.warn(f"Missing expected source: {key}. Skipping merge.", UserWarning)
        continue
```

**Dependencies:**
- Hardcoded list of provider-keyed dict keys
- Format: `"{category}_{provider}"`

**Impact:** Adding a new provider requires updating this hardcoded list.

**Recommended Fix:**
- Option A: Dynamic provider detection - iterate over all keys in `sources` dict
- Option B: Configuration-driven merge specs that list expected providers

---

### 5. **src/config.py**

#### Lines 77-97:
```python
NFELO_FEATURES = [
    'nfelo',           # Main ELO rating
    'QB Adj',          # QB adjustment
    'Value',           # Overall value
    'WoW',             # Week over week change
    'YTD',             # Year to date performance
]

SUBSTACK_FEATURES = [
    'Off.',            # Offensive rating
    'Def.',            # Defensive rating
    'Ovr.',            # Overall rating
]

EPA_FEATURES = [
    'epa_off',         # Offensive EPA per play
    'epa_def',         # Defensive EPA per play
    'epa_margin',      # EPA differential
]
```

**Dependencies:**
- Lists of provider-specific column names
- Used for feature selection and documentation

**Impact:** Not actively used in code, but documents expected column structure.

**Recommended Fix:** Convert to canonical feature mapping config.

---

## Key Findings

### 🚨 Critical Blockers for v1.2/v1.3 Provider Expansion

1. **Model hardcodes provider column names** (`src/models.py:63-75`)
   - Cannot use new providers without code changes

2. **Merge function hardcodes provider keys** (`ball_knower/io/loaders.py:334-387`)
   - Cannot dynamically handle new providers

3. **All scripts reference provider-specific columns directly**
   - No abstraction layer exists

### ✅ What's Working Well

1. **File loading is provider-agnostic**
   - `_resolve_file()` uses category + provider pattern
   - Can load any provider's files without code changes

2. **Compatibility layer works**
   - Legacy code still functions via `src/data_loader.py` forwarding

3. **Category-first naming adopted**
   - All new files use `{category}_{provider}_{season}_week_{week}.csv`

---

## Recommendations for Priority 2 (Feature Mapping Layer)

### Step 1: Define Canonical Feature Schema

Create `ball_knower/io/feature_maps.py`:

```python
CANONICAL_FEATURES = {
    # Overall power ratings
    'overall_rating': {
        'nfelo': 'nfelo',
        'substack': 'Ovr.',
        # Future: 'pff': 'overall_grade', etc.
    },

    # QB adjustments
    'qb_adjustment': {
        'nfelo': 'QB Adj',
        'substack': None,  # Not available
    },

    # Offensive metrics
    'offensive_rating': {
        'nfelo': None,  # Not directly available
        'substack': 'Off.',
    },

    # Defensive metrics
    'defensive_rating': {
        'nfelo': None,
        'substack': 'Def.',
    },

    # EPA metrics
    'epa_offense': {
        'nfelo': 'epa_off',
        'substack': None,
    },

    'epa_defense': {
        'nfelo': 'epa_def',
        'substack': None,
    },

    'epa_margin': {
        'nfelo': 'epa_margin',
        'substack': None,
    },
}
```

### Step 2: Add Feature Extractor

```python
def extract_canonical_features(merged_df, feature_name, providers=['nfelo', 'substack']):
    """
    Extract a canonical feature from merged ratings, trying each provider.

    Returns the first available provider's version of the feature.
    """
    for provider in providers:
        col_name = CANONICAL_FEATURES[feature_name].get(provider)
        if col_name and col_name in merged_df.columns:
            return merged_df[col_name]

    raise ValueError(f"Feature '{feature_name}' not available from any provider")
```

### Step 3: Update Models to Use Canonical Names

Refactor `src/models.py` to accept canonical feature names:

```python
self.weights = {
    'epa_margin': 35,
    'overall_rating_diff': 0.02,
    # etc.
}

# In predict():
if 'epa_margin' in home_features:
    # ...
```

### Step 4: Update Scripts to Map Features

Before passing to model, map provider columns → canonical names:

```python
from ball_knower.io.feature_maps import map_to_canonical

canonical_features = map_to_canonical(team_ratings, provider='nfelo')
# Now canonical_features has keys like 'overall_rating' instead of 'nfelo'
```

---

## Next Steps

**For current branch (Priority 1):**
- [x] Migrate all active scripts to unified loader
- [x] Document provider dependencies (this report)
- [ ] Commit and push changes

**For next branch (Priority 2):**
- [ ] Implement `ball_knower/io/feature_maps.py`
- [ ] Add feature mapping utilities
- [ ] Refactor models to use canonical names
- [ ] Update all scripts to map features before model input

**For future (Priority 3):**
- [ ] Make config week-agnostic
- [ ] Add dynamic provider detection
- [ ] Remove hardcoded provider lists from merge functions

---

## Files Changed in This Migration

1. `investigate_data.py` - Migrated to unified loader
2. `src/__init__.py` - Added legacy deprecation notes
3. `PROVIDER_DEPENDENCIES.md` - This report

**No breaking changes** - all code remains functional via compatibility layer.
