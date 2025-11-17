# BK_Build Project Status Report
**Date:** November 16, 2025
**Current Branch:** `claude/debug-week11-clean-data-01CAQxHgTkdju1Fywvm2cu5q`
**Status:** 🔴 Critical Data Quality Issues Identified

---

## Project Overview

**Goal:** Build an NFL betting model that identifies value bets by predicting Vegas spreads and comparing them to actual lines.

**Approach:** Use nfelo team ratings + historical data to predict what Vegas spread *should be*, then bet when there's significant divergence from actual Vegas lines.

**Data Sources:**
- nflverse: Historical game results, spreads, team stats
- nfelo: Advanced team ratings system
- Week 11 2025: Current week predictions

---

## What We Built

### v1.2 Model (Baseline)
- **Features:** nfelo ratings difference between teams
- **Training:** 2002-2024 seasons
- **Performance:** 1.57 MAE predicting Vegas lines
- **Status:** ✓ Working, committed

### v1.3 Model (EPA Enhanced)
- **Added:** EPA (Expected Points Added) offensive/defensive stats
- **Performance:** 1.63 MAE (slightly worse than v1.2)
- **Status:** ✓ Working, committed

### v1.4 Model (Rolling Stats)
- **Added:** Rolling averages, momentum, interaction terms
- **Performance:** 1.60 MAE (better than v1.3, worse than v1.2)
- **Status:** ✓ Working, committed

---

## What Went Wrong - Week 11 2025 Predictions

### The Problem
When we ran the model on Week 11 2025 games:
- **Expected MAE:** ~1.57 points (vs Vegas lines)
- **Actual MAE:** 13.72 points (vs actual game results)
- **Performance:** 9x worse than expected

### Example: NE vs NYJ
```
Vegas Line:  NE -12.5 (NE favored by 12.5)
Model Pred:  NE -6.88 (NE favored by 6.88)
Edge:        +5.62 points
Model Bet:   NYJ +12.5 (away team)
Actual:      NE won by 13 → Vegas was right, model was very wrong
```

---

## Root Causes Identified

### 1. **DUPLICATE DATA** 🔴
nfelo ratings snapshot has duplicate entries:
```
Team  First Rating  Second Rating  Difference
NE    1519.8        1516.8         3.0 points
NYJ   1396.3        1392.3         4.0 points
```
**Impact:** Model uses wrong rating (just takes first occurrence)

### 2. **MISSING TEAM MAPPINGS** 🔴
```
Week 11 uses: "LA" and "LV"
nfelo uses:   "LAR" (not "LA")
Missing:      LA, LV
```
**Impact:** Some teams get NaN ratings or wrong team ratings

### 3. **STALE DATA** 🟡
nfelo ratings may not reflect:
- Recent injuries (Aaron Rodgers struggling)
- Team chaos (NYJ fired their coach mid-season)
- 2025 mid-season context
**Impact:** Model doesn't know what Vegas knows

### 4. **BETTING LOGIC CONFUSION** 🟡
```
Edge = Model - Vegas
Positive Edge = Model predicts bigger spread than Vegas
BUT: Code was recommending "Bet Home" for positive edge
SHOULD BE: "Bet Away" (fade the favorite)
```
**Impact:** All betting recommendations were inverted

---

## Critical Next Steps

### IMMEDIATE (Data Cleaning)
1. **Fix duplicate nfelo ratings** - Identify and remove duplicates
2. **Fix team name mapping** - Handle LA/LAR, add LV, ensure all teams map correctly
3. **Validate data integrity** - Build automated checks for:
   - Duplicate teams
   - Missing teams
   - Rating sanity checks (should be ~1200-1700)
   - Chronological ordering
   - Data freshness

### SECONDARY (Model Improvements)
1. Add injury/QB data
2. Add recent performance trends (last 3-5 games)
3. Consider predicting outcomes instead of Vegas lines
4. Add confidence intervals / bet sizing

---

## Current File Structure

```
BK_Build/
├── src/nflverse_data.py         # Data loading utilities
├── ball_knower_v1_2.py          # Baseline model (nfelo only)
├── ball_knower_v1_3.py          # + EPA features
├── ball_knower_v1_4.py          # + Rolling averages
├── predict_current_week.py      # Week 11 predictions (BROKEN)
│
├── Diagnostic Scripts (NEW)
├── investigate_spreads.py       # Analyzes spread conventions
├── check_model_accuracy.py      # Compares model vs actual results
├── diagnose_model_failure.py    # Identifies data quality issues
└── WEEK_11_ISSUES_ANALYSIS.md   # Full technical analysis
```

---

## Key Insights

1. **The model works conceptually** - 1.57 MAE vs Vegas during training is good
2. **Garbage in, garbage out** - Data quality issues make predictions unreliable
3. **Vegas knows more** - Vegas incorporates injuries, coaching, context the model doesn't have
4. **Need validation framework** - Can't trust predictions without validating input data

---

## What Needs to Happen Now

### Phase 1: Data Cleaning & Validation ⚠️ CRITICAL
- Build automated data validation checks
- Clean duplicate nfelo ratings
- Fix team name mapping issues
- Verify data freshness and accuracy
- **Goal:** Trust the input data before trusting model outputs

### Phase 2: Re-run & Validate
- Re-run Week 11 predictions with clean data
- Compare to actual results
- Verify betting logic is correct
- Assess if model actually has edge

### Phase 3: Model Enhancement (if Phase 2 shows promise)
- Add injury data
- Add recent trends
- Add QB ratings
- Consider ensemble approaches

---

## Questions to Answer

1. **Can we get clean, up-to-date nfelo ratings for Week 11 2025?**
2. **Should we scrape real-time data instead of using snapshots?**
3. **Do we need a different data source entirely?**
4. **Is predicting Vegas lines the right approach, or should we predict outcomes?**

---

## Bottom Line

We built a model that works in theory (good training performance), but failed in practice due to **data quality issues**. We need to:

1. **STOP** making predictions until data is clean
2. **BUILD** a robust data validation framework
3. **FIX** duplicate ratings, team mappings, data freshness
4. **VERIFY** the cleaned data produces sensible predictions
5. **THEN** decide if the model has real betting value

**The code is fine. The data is broken.** 🔧
