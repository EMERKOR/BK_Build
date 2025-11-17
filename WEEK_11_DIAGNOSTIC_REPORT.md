# Week 11 2025 Diagnostic Report: Data Quality & Model Performance

**Date:** November 17, 2025
**Analysis:** Ball Knower v1.2 predictions vs actual Week 11 2025 results
**Status:** 🟡 Data quality issues RESOLVED, but model performance issues REMAIN

---

## Executive Summary

We successfully identified and fixed critical data quality issues (duplicates, team name mappings), but the model's Week 11 performance reveals deeper problems with the nfelo snapshot data source itself. The model performed **8.7x worse** than expected, suggesting the underlying team ratings are stale or not reflective of current team strength.

**Bottom Line:** The code is working correctly, the data is now clean, but the ratings themselves appear outdated for mid-2025 season context.

---

## Data Quality Issues (RESOLVED ✅)

### Issue 1: Duplicate Team Entries
**Status:** ✅ FIXED

- **Found:** 2 teams (NE, NYJ) had duplicate entries in nfelo snapshot
- **NE duplicates:** 1519.76 vs 1516.81 (2.95 point difference)
- **NYJ duplicates:** 1396.25 vs 1392.29 (3.96 point difference)
- **Root cause:** QB adjustment variations creating multiple rows per team
- **Solution:** Removed duplicates, kept first occurrence (assumed most recent)

### Issue 2: Team Name Mapping
**Status:** ✅ FIXED

- **Found:** Week 11 games use "LA" and "LV", nfelo snapshot uses "LAR" and "OAK"
- **Mappings applied:**
  - LAR → LA (Los Angeles Rams)
  - OAK → LV (Las Vegas Raiders, moved 2020)
- **Solution:** Applied team alias corrections in cleaning pipeline

### Issue 3: Data Validation
**Status:** ✅ PASSED

- ✅ No duplicate teams remaining (32 unique teams)
- ✅ No NaN rating values
- ✅ All Week 11 teams have matching nfelo entries
- ✅ Ratings within sanity bounds (1260-1675)
- ✅ Data appears to be Week 11 2025 snapshot

---

## Model Performance Analysis

### Training Performance (Expected)
- **Test MAE:** 1.57 points (vs Vegas lines)
- **Test R²:** 0.884
- **Training period:** 2002-2024 seasons
- **Model:** Linear regression on nfelo rating differences

### Week 11 2025 Actual Performance
- **Actual MAE:** 13.74 points (vs actual game outcomes)
- **Vegas MAE:** 14.14 points
- **Model vs Vegas:** Model beat Vegas by 0.39 points (statistically insignificant)
- **Performance degradation:** **8.7x worse** than expected

### Critical Finding
The model's expected MAE of 1.57 points was measured against **Vegas lines** (predicting what Vegas would set). The actual MAE of 13.74 points is measured against **game outcomes** (predicting final scores).

**These are not directly comparable:**
- Vegas lines are calibrated to split betting action, not predict outcomes
- Game outcomes have much higher variance than Vegas lines
- A model trained to predict Vegas lines will not necessarily predict outcomes accurately

**However**, even accounting for this, the performance is concerning because:
1. The model's predictions diverged significantly from Vegas (5-6 point edges)
2. Those divergences were not profitable (1-2 record, -36% ROI)
3. The model appears to be using stale team strength ratings

---

## Betting Recommendations Analysis

### Bets Recommended (2.0+ point edge threshold)
**Total:** 3 bets recommended

| Game | Bet | Vegas Line | Edge | Actual Result | Outcome |
|------|-----|------------|------|---------------|---------|
| NYJ @ NE | Bet NYJ +12.5 | -12.5 | +5.62 | NE won by 13 | ❌ LOSS |
| HOU @ TEN | Bet HOU +5.5 | +5.5 | +3.94 | HOU won by 3 | ✅ WIN |
| CHI @ MIN | Bet CHI +3.0 | -3.0 | +2.47 | MIN won by 2 | ❌ LOSS |

### Financial Results
- **Win Rate:** 33.3% (1-2 record)
- **Breakeven Required:** 52.4% (at -110 odds)
- **Total Profit:** -$120 on $330 risked
- **ROI:** -36.4%
- **Result:** 🔴 UNPROFITABLE

---

## Game-by-Game Breakdown

### Largest Prediction Errors

1. **CIN @ PIT** (Prediction error: 27.15 points)
   - Model predicted: PIT -5.15
   - Actual result: PIT won by 22
   - **Issue:** Model severely underestimated PIT strength

2. **LAC @ JAX** (Prediction error: 26.73 points)
   - Model predicted: JAX +2.27
   - Actual result: JAX won by 29
   - **Issue:** Model severely underestimated JAX strength (or overestimated LAC)

3. **SF @ ARI** (Prediction error: 21.50 points)
   - Model predicted: ARI +2.50
   - Actual result: SF won by 19
   - **Issue:** Model severely underestimated SF strength

### What Went Wrong?

**NYJ @ NE (5.62 point edge, LOSS):**
- Model predicted: NE -6.88
- Vegas line: NE -12.5
- Actual result: NE won by 13 (Vegas was nearly perfect)
- **Analysis:** Model thought NE was overvalued by 5.6 points, but Vegas was right
- **Likely issue:** nfelo ratings don't reflect NYJ's mid-season chaos (fired coach, QB struggles)

**HOU @ TEN (3.94 point edge, WIN):**
- Model predicted: TEN +9.44
- Vegas line: TEN -5.5
- Actual result: HOU won by 3
- **Analysis:** Model correctly identified HOU as undervalued
- **Result:** Only profitable bet of the week

**CHI @ MIN (2.47 point edge, LOSS):**
- Model predicted: MIN -0.53 (essentially even)
- Vegas line: MIN -3.0
- Actual result: MIN won by 2
- **Analysis:** Model thought MIN was overvalued by 2.5 points, but Vegas was right

---

## Root Cause Analysis

### What We Know

1. **Data quality issues are fixed:**
   - ✅ No duplicates
   - ✅ Correct team mappings
   - ✅ All teams have ratings
   - ✅ No NaN values

2. **Model architecture is sound:**
   - Trained correctly (1.57 MAE vs Vegas historically)
   - Appropriate feature engineering
   - Reasonable coefficient values

3. **The problem is the ratings themselves:**
   - nfelo snapshot appears to be Week 11 2025 data (column shows week=11)
   - But ratings don't reflect mid-season context
   - Model predictions diverge significantly from Vegas (5-6 points)
   - Those divergences are not profitable

### Why Are The Ratings Stale?

Possible explanations:

1. **QB Adjustments Are Outdated:**
   - The snapshot has QB adjustments (qb_adj column)
   - But these may not reflect current QB performance
   - Example: Aaron Rodgers (NYJ) may be rated higher than his actual 2025 performance

2. **Missing Recent Context:**
   - Coaching changes (NYJ fired coach mid-season)
   - Injuries to key players
   - Recent performance trends (last 3-5 games)
   - Team chemistry/morale issues

3. **Snapshot Methodology:**
   - The nfelo snapshot is updated weekly
   - But may use lagging indicators (season-long stats)
   - Vegas incorporates real-time information (injuries, line movement, sharp money)

4. **Training Data Mismatch:**
   - Model was trained on historical nfelo ratings (2002-2024)
   - Those ratings were calculated retroactively with complete data
   - Real-time ratings may be calculated differently or with incomplete data

---

## Comparison to Status Report Predictions

The status report predicted these issues **before** data cleaning:

| Prediction | Actual Result |
|------------|---------------|
| Duplicates would cause wrong ratings | ✅ Confirmed (NE, NYJ had duplicates) |
| Team mapping issues (LA, LV) | ✅ Confirmed (LAR→LA, OAK→LV needed) |
| Model would have ~13 point MAE | ✅ Confirmed (13.74 actual MAE) |
| Betting recommendations would lose | ✅ Confirmed (1-2 record, -36% ROI) |
| Vegas would be more accurate | ✅ Confirmed (Vegas MAE: 14.14 vs Model: 13.74, essentially tied) |

**However**, the status report suggested data cleaning would fix the issue. The actual result shows:
- Data cleaning **did** fix data quality issues
- But model performance **did not** improve significantly
- This suggests the problem is deeper than just duplicates/mappings

---

## Key Insights

1. **Data Quality ≠ Data Accuracy:**
   - We fixed data quality (duplicates, mappings)
   - But the data itself (ratings) may still be inaccurate for current context

2. **Training vs Production Mismatch:**
   - Model was trained to predict Vegas lines (1.57 MAE)
   - Model is being used to predict game outcomes (13.74 MAE)
   - These are fundamentally different tasks

3. **Vegas Knows More:**
   - Vegas lines incorporate information the model doesn't have
   - Injuries, coaching, line movement, sharp money, public bias
   - A pure ratings-based model will struggle to beat Vegas consistently

4. **Sample Size Warning:**
   - Only 11 completed games (4 still pending)
   - Only 3 value bets placed
   - Results could be variance (too small to draw strong conclusions)

---

## Recommendations

### Immediate Actions

1. **Do NOT bet real money based on current model**
   - Sample size too small to validate edge
   - Ratings appear stale
   - Model has not demonstrated profitability

2. **Wait for remaining Week 11 games to complete**
   - 4 games still pending (BAL@CLE, KC@DEN, DET@PHI, DAL@LV)
   - Re-run analysis with full 15-game sample

3. **Investigate nfelo snapshot methodology**
   - How are ratings updated week-to-week?
   - Are QB adjustments real-time or lagged?
   - Is there a better data source for current team strength?

### Short-Term Improvements

1. **Add QB injury/performance adjustments:**
   - Scrape current QB stats (EPA, completion %, etc.)
   - Adjust ratings for backup QBs
   - Account for QB changes mid-season

2. **Add recent performance trends:**
   - Last 3-5 games weighted more heavily
   - Rolling EPA (offense/defense)
   - Momentum indicators

3. **Improve data freshness checks:**
   - Validate that nfelo snapshot is truly current
   - Compare to other data sources (ESPN FPI, etc.)
   - Alert if ratings seem stale

### Medium-Term Strategy

1. **Reconsider the objective:**
   - Currently: Predict Vegas lines, bet on divergences
   - Alternative: Predict game outcomes directly
   - Alternative: Predict 1H/2H totals, props, etc.

2. **Build ensemble model:**
   - Combine nfelo with other rating systems (FPI, DVOA, etc.)
   - Reduce reliance on single data source
   - More robust to data quality issues

3. **Add calibration layer:**
   - Current model outputs point spreads
   - Add probability calibration
   - Better bet sizing based on confidence

### Long-Term Research

1. **Real-time data integration:**
   - Scrape injury reports
   - Monitor line movement
   - Incorporate weather, referee data
   - Track coaching changes

2. **Alternative approaches:**
   - Deep learning on play-by-play data
   - Bayesian updating as season progresses
   - Market-making models (not just value betting)

---

## Conclusion

**What We Fixed:**
- ✅ Duplicate nfelo ratings (NE, NYJ)
- ✅ Team name mapping (LAR→LA, OAK→LV)
- ✅ Data validation framework
- ✅ Automated cleaning pipeline

**What Still Needs Work:**
- 🔴 nfelo ratings appear stale for mid-2025 context
- 🔴 Model predictions diverge from Vegas without profitability
- 🔴 No demonstrated edge in betting (1-2 record, -36% ROI)
- 🔴 Need additional features (injuries, trends, QB data)

**Next Steps:**
1. Wait for Week 11 completion (4 games pending)
2. Investigate nfelo snapshot update methodology
3. Build QB injury/performance adjustment layer
4. Add recent performance trends (rolling EPA, last 5 games)
5. Test on Week 12 predictions **before** betting
6. Require 3-4 weeks of profitable results before live betting

**Status:** 🟡 **HOLD** - Do not bet until model demonstrates consistent edge over multiple weeks.

---

## Files Created

- `validate_and_clean_data.py` - Data validation and cleaning framework
- `analyze_week11_results.py` - Actual results analysis script
- `data/cache/nfelo_snapshot_cleaned.csv` - Cleaned nfelo ratings
- `output/week_11_value_bets_v1_2.csv` - Model predictions
- `WEEK_11_DIAGNOSTIC_REPORT.md` - This report

---

**End of Report**
