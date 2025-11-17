# Post-v1.2 Merge Roadmap

**Current State:** v1.2 branch ready for validation and merge
**Target:** Plan for v1.3+ implementation
**Date:** 2025-11-17

---

## Executive Summary

This roadmap outlines the recommended next steps after successfully validating and merging the v1.2 unified loader architecture into main. It provides a clear path for:

1. **v1.2 Post-Merge Cleanup** - Immediate tasks after merge
2. **v1.3 Implementation Planning** - When and how to build market feature integration
3. **Data Preparation Strategy** - Isolated scripts for market data ingestion
4. **Feature Development Order** - Proper sequence for v1.3 features
5. **Long-Term Architecture** - Scaling to v1.4+ with Next Gen Stats

---

## Phase 1: Post-v1.2 Merge Cleanup (Week 1)

**Trigger:** After v1.2 validation passes and merge to main is complete

### 1.1 Immediate Post-Merge Tasks

#### Archive Documentation
- [ ] **Create ARCHIVE.md**
  - Document all deprecated scripts
  - List `ball_knower_v1_1.py`, `bk_v1_final.py`, `calibrate_*.py`, etc.
  - Add "DO NOT USE" warnings
  - Provide migration path to v1.2

- [ ] **Update README.md**
  - Add "Quick Start with v1.2" section
  - Show unified loader usage examples
  - Link to `MERGE_VALIDATION_CHECKLIST.md` for validation history
  - Update model version table (v1.2 is now canonical)

#### Tag and Release
- [ ] **Create release tag**
  ```bash
  git tag -a v1.2-unified-loader -m "v1.2: Unified loader architecture + category-first naming"
  git push origin v1.2-unified-loader
  ```

- [ ] **Create GitHub release**
  - Release notes highlighting unified loader benefits
  - Breaking changes: file naming convention (legacy fallback works but deprecated)
  - Migration guide for users

#### Dependency Audit
- [ ] **Run dependency check**
  ```bash
  pip list --outdated
  ```
  - Update `requirements.txt` if needed
  - Ensure `scikit-learn`, `pandas`, `numpy` are current

- [ ] **Clean up unused imports**
  - Run linter on active scripts
  - Remove unused legacy imports from v1.2 scripts

---

## Phase 2: v1.3 Planning & Preparation (Week 1-2)

**Trigger:** After Phase 1 cleanup is complete

### 2.1 When to Begin v1.3 Implementation

**Recommendation: Start v1.3 immediately after Phase 1 cleanup**

**Rationale:**
- v1.2 provides stable foundation (unified loader + historical data)
- v1.3 builds on v1.2 by adding **market-driven features**
- No architectural conflicts between v1.2 and v1.3
- Can develop v1.3 in parallel with v1.2 production use

**Development Branch:**
```bash
git checkout main
git pull origin main
git checkout -b feature/v1.3-market-integration
```

### 2.2 v1.3 Feature Scope (From Design Draft)

**Core Enhancement: Market Signal Integration**

v1.3 adds these feature categories:
1. **Line Movement Features**
   - Opening vs. current spread delta
   - Line movement velocity (pts/hour)
   - Cross-book consensus vs. outlier detection

2. **Market Context Features**
   - Public betting % (consensus vs. sharp money)
   - Implied probability from odds
   - Historical CLV (Closing Line Value) for teams

3. **Rolling Performance Features**
   - 3-game rolling EPA offense/defense
   - 5-game rolling point differential vs. expectations
   - Rest-adjusted performance metrics

**Model Architecture:**
- Extend v1.2 Ridge regression with new feature set
- Target: Improve MAE from 1.57 → 1.35 points
- Maintain time-series cross-validation (no look-ahead bias)

---

## Phase 3: Data Preparation Strategy (Week 2)

**Trigger:** Before v1.3 model implementation

### 3.1 Isolated Data Prep Scripts

**Recommendation: Create separate data ingestion pipeline for market features**

**Rationale:**
- Market data changes frequently (unlike historical stats)
- Need separate validation/monitoring
- Don't pollute unified loader with volatile market data
- Allow independent testing of market data quality

#### 3.1.1 Create Market Data Scripts

**Script 1: `ingest_market_lines.py`**
```python
"""
Fetch and normalize betting market data from multiple sportsbooks.

Outputs: data/market_lines/lines_{season}_week_{week}.csv
"""
```

**Purpose:**
- Fetch opening/current spreads from DraftKings, FanDuel, BetMGM
- Calculate consensus spread
- Detect line movement
- Store in `data/market_lines/` (separate from `data/current_season/`)

**Data Source Options:**
- The Odds API (https://the-odds-api.com/) - paid but reliable
- ESPN Bet API - if available
- Manual scraping (DraftKings, FanDuel) - last resort

---

**Script 2: `ingest_public_betting_trends.py`**
```python
"""
Fetch public betting percentages and sharp money indicators.

Outputs: data/betting_trends/trends_{season}_week_{week}.csv
"""
```

**Purpose:**
- Get public betting % (e.g., from Action Network, Covers.com)
- Track reverse line movement (RLM) signals
- Store in `data/betting_trends/`

**Data Source Options:**
- Action Network API (if available)
- Vegas Insider scraping
- Manual weekly collection (interim solution)

---

**Script 3: `calculate_rolling_features.py`**
```python
"""
Calculate rolling performance metrics from historical data.

Inputs: nfelo historical games, EPA data
Outputs: data/rolling_features/rolling_{season}_week_{week}.csv
"""
```

**Purpose:**
- 3-game rolling EPA (offense/defense)
- 5-game rolling point differential vs. expectation
- Rest-adjusted metrics
- Store in `data/rolling_features/`

**Data Sources:**
- nfelo historical games (already used in v1.2)
- EPA data from nflverse

---

#### 3.1.2 Market Data Loader Extension

**Option A: Extend Unified Loader (Recommended)**
```python
# In ball_knower/io/loaders.py

def load_market_lines(season: int, week: int, data_dir=None):
    """Load betting market lines and movement data."""
    path = _resolve_file("market_lines", "aggregated", season, week, data_dir)
    return pd.read_csv(path)

def load_betting_trends(season: int, week: int, data_dir=None):
    """Load public betting % and sharp money indicators."""
    path = _resolve_file("betting_trends", "aggregated", season, week, data_dir)
    return pd.read_csv(path)

def load_rolling_features(season: int, week: int, data_dir=None):
    """Load rolling performance features."""
    path = _resolve_file("rolling_features", "computed", season, week, data_dir)
    return pd.read_csv(path)
```

**Option B: Separate Market Loader Module**
```python
# Create ball_knower/io/market_loaders.py for market-specific data
# Keep ball_knower/io/loaders.py for stable rating data
```

**Recommendation: Use Option A** - Extend unified loader
- Maintains consistency
- Reuses file resolution logic
- Easier to merge market + rating data for v1.3

---

### 3.2 Data Directory Structure

**Proposed structure after v1.3 data prep:**
```
data/
├── current_season/          # v1.2 stable rating data
│   ├── power_ratings_nfelo_2025_week_11.csv
│   ├── epa_tiers_nfelo_2025_week_11.csv
│   └── ...
├── market_lines/            # NEW: Market data
│   ├── lines_2025_week_11.csv
│   └── ...
├── betting_trends/          # NEW: Public betting data
│   ├── trends_2025_week_11.csv
│   └── ...
├── rolling_features/        # NEW: Computed rolling stats
│   ├── rolling_2025_week_11.csv
│   └── ...
└── historical/              # Existing: Historical archives
    └── ...
```

**Benefits:**
- Clear separation of stable vs. volatile data
- Easy to add new market data sources
- Can version control `current_season/` but exclude `market_lines/` (changes hourly)

---

## Phase 4: v1.3 Implementation Order (Week 3-4)

**Trigger:** After market data scripts are tested and producing data

### 4.1 Correct Build Order

**Step 1: Feature Ingestion (Week 3, Days 1-3)**
- [ ] Implement `ingest_market_lines.py`
- [ ] Implement `ingest_public_betting_trends.py`
- [ ] Implement `calculate_rolling_features.py`
- [ ] Test data quality for Week 11, 2025
- [ ] Validate schema: all required columns present

**Step 2: Loader Extension (Week 3, Days 4-5)**
- [ ] Add `load_market_lines()` to unified loader
- [ ] Add `load_betting_trends()` to unified loader
- [ ] Add `load_rolling_features()` to unified loader
- [ ] Update `load_all_sources()` to include v1.3 data
- [ ] Test loader with Week 11 data

**Step 3: Feature Engineering (Week 4, Days 1-2)**
- [ ] Create `src/market_features.py` module
- [ ] Implement line movement calculations
- [ ] Implement market context features
- [ ] Implement rolling performance features
- [ ] Unit test feature calculations

**Step 4: Model Training (Week 4, Days 3-4)**
- [ ] Create `ball_knower_v1_3.py` training script
- [ ] Extend v1.2 feature set with v1.3 features
- [ ] Train on historical data (2009-2024)
- [ ] Time-series cross-validation
- [ ] Evaluate on 2025 test set

**Step 5: Backtesting (Week 4, Day 5)**
- [ ] Create `backtest_v1_3.py`
- [ ] Compare v1.3 vs. v1.2 performance
- [ ] Measure improvement in MAE, R², CLV
- [ ] Document results in `V1.3_RESULTS.md`

**Step 6: Production Deployment (Week 5, Day 1)**
- [ ] Create `predict_current_week_v1_3.py`
- [ ] Test on Week 11, 2025
- [ ] Compare predictions to v1.2
- [ ] Deploy if v1.3 beats v1.2 on test metrics

---

### 4.2 Why This Order Matters

**✓ Feature Ingestion → Validation → Modeling**
- **Correct:** Ensures data quality before building features
- **Avoids:** Training model with corrupt/incomplete data

**✗ Modeling → Feature Ingestion** (Wrong order)
- Risk: Mock data during development, real data breaks in production
- Risk: Schema mismatches discovered too late

**✓ Loader Extension → Feature Engineering**
- **Correct:** Standardized data access, easier testing
- **Avoids:** Ad-hoc file reading scattered across code

---

## Phase 5: v1.3 Validation & Merge (Week 5)

### 5.1 Pre-Merge Checklist

- [ ] **Model performance exceeds v1.2**
  - Test MAE < 1.57 (v1.2 baseline)
  - Test R² > 0.884 (v1.2 baseline)
  - CLV analysis shows positive expected value

- [ ] **Data pipeline is reliable**
  - Market data scripts run without errors
  - Data freshness checks (no stale data)
  - Fallback logic if market API is down

- [ ] **Code quality**
  - All tests pass
  - Linting passes
  - Documentation updated

- [ ] **Backward compatibility**
  - v1.2 scripts still work (don't break v1.2 in production)
  - Users can opt-in to v1.3 features

### 5.2 Merge Strategy

**Option A: Feature Flag (Recommended)**
```python
# In src/config.py
USE_V1_3_FEATURES = False  # Set to True to enable v1.3

# In ball_knower/io/loaders.py
def load_all_sources(season, week, use_v1_3=USE_V1_3_FEATURES):
    result = {...}  # Load v1.2 data
    if use_v1_3:
        result['market_lines'] = load_market_lines(season, week)
        result['betting_trends'] = load_betting_trends(season, week)
        result['rolling_features'] = load_rolling_features(season, week)
    return result
```

**Benefits:**
- Deploy v1.3 code to main without forcing everyone to use it
- A/B test v1.2 vs. v1.3 in production
- Easy rollback if v1.3 has issues

**Option B: Separate v1.3 Branch (Not Recommended)**
- Harder to maintain
- Divergence between v1.2 and v1.3 codebases
- Only use if v1.3 requires breaking changes

---

## Phase 6: v1.4+ Long-Term Planning (Future)

### 6.1 v1.4 Scope (Next Gen Stats Integration)

**Note:** v1.4 design already exists in `claude/review-model-features-*` branch

**Features:**
- Player tracking data (separation, route metrics)
- Advanced QB metrics (completion probability over expected)
- Pressure rates, coverage metrics

**Timeline:** Start v1.4 after v1.3 is in production (Week 6+)

**Data Sources:**
- Next Gen Stats API (nfl.com)
- Pro Football Focus (PFF) - if accessible
- TruMedia (if subscription available)

### 6.2 Architecture Evolution

**v1.2 → v1.3 → v1.4 Feature Layers:**

```
┌─────────────────────────────────────┐
│  v1.4: Next Gen Stats (Player)     │  ← Most granular, highest signal
├─────────────────────────────────────┤
│  v1.3: Market Features (Game)      │  ← Market efficiency signals
├─────────────────────────────────────┤
│  v1.2: Historical Ratings (Team)   │  ← Stable foundation
└─────────────────────────────────────┘
```

**Key Principle:**
- Each layer builds on previous
- Can deploy incrementally (v1.2 works standalone, v1.3 optional, v1.4 optional)
- Unified loader handles all data sources

### 6.3 Model Complexity Management

**Current: Ridge Regression (v1.2)**
- Simple, interpretable
- Fast training
- Good baseline

**v1.3: Ridge + Feature Selection**
- May need feature selection (too many features = overfitting)
- Consider LASSO for automatic feature selection
- Still linear, still interpretable

**v1.4: Potential Upgrade to Gradient Boosting**
- More features = more interactions
- XGBoost/LightGBM can capture non-linear relationships
- Trade-off: Less interpretable, but higher accuracy
- **Decision point:** Only if v1.3 hits diminishing returns with linear models

---

## Phase 7: Additional Cleanup Tasks

### 7.1 Testing Infrastructure

**Current State:** Manual testing via scripts

**Recommendation: Add pytest suite**

**Create `tests/` directory:**
```
tests/
├── test_loaders.py          # Test unified loader functions
├── test_market_features.py  # Test v1.3 feature engineering
├── test_models.py           # Test model predictions
└── conftest.py              # Shared fixtures
```

**Example test:**
```python
# tests/test_loaders.py
import pytest
from ball_knower.io import loaders

def test_load_power_ratings_nfelo():
    df = loaders.load_power_ratings("nfelo", 2025, 11)
    assert "team" in df.columns
    assert len(df) == 32  # All NFL teams
    assert df["nfelo"].notna().all()
```

**Benefits:**
- Catch regressions early
- Document expected behavior
- CI/CD integration (GitHub Actions)

### 7.2 Documentation Improvements

- [ ] **Add docstrings to all loader functions**
  - Already done in v1.2, verify completeness

- [ ] **Create CONTRIBUTING.md**
  - How to add new data sources
  - How to add new model versions
  - Code style guide

- [ ] **Add usage examples to README.md**
  - Quick start with v1.2
  - How to run backtests
  - How to generate predictions

### 7.3 CI/CD Pipeline (Future)

**GitHub Actions workflows:**
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/
```

**Benefits:**
- Automated testing on every commit
- Catch breaking changes before merge
- Enforce code quality

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Post-v1.2 Cleanup** | Week 1 | Archive docs, release tag, dependency audit |
| **Phase 2: v1.3 Planning** | Week 1-2 | Data source research, feature design validation |
| **Phase 3: Data Prep** | Week 2 | Market data scripts, loader extensions |
| **Phase 4: v1.3 Implementation** | Week 3-4 | Feature engineering, model training, backtesting |
| **Phase 5: v1.3 Validation** | Week 5 | Testing, merge to main with feature flag |
| **Phase 6: v1.4 Planning** | Week 6+ | Next Gen Stats integration (future) |

**Critical Path:**
```
v1.2 Merge ➔ Cleanup ➔ Market Data Scripts ➔ v1.3 Features ➔ v1.3 Model ➔ Validation ➔ Deploy
```

---

## Decision Points

### Decision 1: Market Data Source
**When:** Phase 3, Week 2
**Options:**
- A) The Odds API (paid, reliable, $50-200/mo)
- B) Manual scraping (free, fragile, time-consuming)
- C) Hybrid (API for opening lines, manual for public %)

**Recommendation:** Start with Option C (hybrid)
- Use The Odds API for line data (worth the cost)
- Manually collect public % from Vegas Insider weekly (until API found)

### Decision 2: v1.3 Loader Architecture
**When:** Phase 3, Week 2
**Options:**
- A) Extend unified loader (`ball_knower/io/loaders.py`)
- B) Separate market loader module

**Recommendation:** Option A (extend unified loader)
- See Section 3.1.2 for rationale

### Decision 3: v1.3 Deployment Strategy
**When:** Phase 5, Week 5
**Options:**
- A) Feature flag (v1.3 opt-in)
- B) Separate branch
- C) Replace v1.2 entirely

**Recommendation:** Option A (feature flag)
- See Section 5.2 for rationale
- Allows A/B testing in production

### Decision 4: When to Start v1.4
**When:** After v1.3 in production
**Trigger:** v1.3 shows consistent improvement over v1.2 for 4+ weeks

**Don't start v1.4 if:**
- v1.3 is unstable
- Market data pipeline is unreliable
- Team capacity is limited

---

## Risk Mitigation

### Risk 1: Market Data API Unavailable
**Likelihood:** Medium
**Impact:** High (blocks v1.3)

**Mitigation:**
- Identify 2-3 backup data sources during Phase 2
- Build fallback logic (use v1.2 if market data missing)
- Manual collection as last resort

### Risk 2: v1.3 Doesn't Improve on v1.2
**Likelihood:** Low-Medium
**Impact:** Medium (wasted effort, but learning)

**Mitigation:**
- Validate features on historical data before full implementation
- Small-scale backtest (single season) before full training
- Feature importance analysis (drop low-signal features)

### Risk 3: Over-Engineering
**Likelihood:** Medium
**Impact:** Low-Medium (complexity slows future work)

**Mitigation:**
- Follow "build what you need" principle
- Don't add features speculatively
- Keep v1.2 as production fallback

---

## Success Metrics

### v1.2 Success (Now)
- ✓ Unified loader works
- ✓ All scripts use canonical API
- ✓ No breaking changes in production

### v1.3 Success (Week 5)
- Test MAE < 1.50 (vs. v1.2: 1.57)
- Test R² > 0.90 (vs. v1.2: 0.884)
- Positive CLV on backtests (beating closing line consistently)

### Long-Term Success (3-6 months)
- v1.4 with Next Gen Stats deployed
- Automated data pipeline (no manual collection)
- Testing infrastructure (CI/CD with pytest)
- 3+ model versions maintained (v1.2, v1.3, v1.4)

---

## Appendix A: Quick Reference Commands

### Validate v1.2 Merge
```bash
# Run full validation checklist
python test_data_loading.py
python run_demo.py
python ball_knower_v1_2.py
```

### Start v1.3 Development
```bash
git checkout main
git pull origin main
git checkout -b feature/v1.3-market-integration
```

### Create Market Data Scripts
```bash
mkdir -p data/{market_lines,betting_trends,rolling_features}
touch ingest_market_lines.py
touch ingest_public_betting_trends.py
touch calculate_rolling_features.py
```

### Run v1.3 Training (Future)
```bash
python ball_knower_v1_3.py              # Train model
python backtest_v1_3.py                 # Validate performance
python predict_current_week_v1_3.py     # Generate predictions
```

---

## Appendix B: Related Documents

- **v1.2 Validation:** `MERGE_VALIDATION_CHECKLIST.md` (this branch)
- **v1.2 Report:** `V1.2_MERGE_READINESS_REPORT.md` (previous session)
- **v1.3 Design:** `V1.3_DESIGN_DRAFT.md` (previous session)
- **v1.4 Design:** Available in `claude/review-model-features-*` branch
- **Data Setup:** `DATA_SETUP_GUIDE.md`
- **Step 2 Assessment:** `STEP_2_ASSESSMENT.md`

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Next Review:** After v1.2 merge completion
