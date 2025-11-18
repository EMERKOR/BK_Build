# Ball Knower Model Overview

**Last Updated:** 2025-11-18
**Purpose:** Comprehensive audit of all modeling, training, and backtest artifacts in the BK_Build repository

This document provides a detailed analysis of each modeling script, including what it predicts, how it's evaluated, and where Vegas lines enter the system.

---

## Table of Contents

1. [Main Model Scripts](#main-model-scripts)
2. [Backtest Scripts](#backtest-scripts)
3. [Calibration Scripts](#calibration-scripts)
4. [Summary Table](#summary-table)
5. [Objective Misalignment and Risks](#objective-misalignment-and-risks)

---

## Main Model Scripts

### ball_knower_v1_final.py (v1.0)

**Location:** `/home/user/BK_Build/ball_knower_v1_final.py:1`

**Purpose:** Production baseline model using deterministic formula

**Target Variable:**
- **None** (deterministic model - no training)
- Predictions are compared against `spread_line` (Vegas closing line) for evaluation only

**Input Features:**
- `nfelo_diff` (home_nfelo - away_nfelo)
- Home field advantage constant (2.67 points)

**Data Range:**
- Applies to Week 11, 2025 (current week predictions)
- Calibration coefficients derived from Week 11 Vegas lines (14 games)

**Evaluation Metrics:**
- MAE (Mean Absolute Error) vs Vegas lines: `ball_knower_v1_final.py:91`
- RMSE vs Vegas lines: `ball_knower_v1_final.py:92`
- Edge = `bk_v1_spread - spread_line` (difference from Vegas): `ball_knower_v1_final.py:71`

**What is `bk_line`?**
- `bk_v1_spread = INTERCEPT + (nfelo_diff * NFELO_COEF)`: `ball_knower_v1_final.py:70`
- This is meant to be an **independent football-based prediction**
- HOWEVER, coefficients (0.0447, 2.67) were calibrated to minimize error vs Vegas on Week 11

**Where does Vegas enter?**
- As **comparison benchmark** for evaluation: `ball_knower_v1_final.py:71`
- Coefficients were **fitted to Vegas** during calibration (see calibrate_regression.py)
- "Edge" is defined as difference from Vegas, not from actual outcomes

**Key Finding:**
- Although labeled as "deterministic baseline," the coefficients were derived by fitting to Vegas lines, not football theory or actual game outcomes

---

### ball_knower_v1_1.py (v1.1)

**Location:** `/home/user/BK_Build/ball_knower_v1_1.py:1`

**Purpose:** Enhanced situational model with rest, divisional, and surface adjustments

**Target Variable:**
- **None** (deterministic model - no training)
- Adjustments are hand-calibrated from nfelo ELO point analysis

**Input Features:**
- Base: `nfelo_diff` × 0.0447 + 2.67 HFA (from v1.0)
- Rest advantage: bye weeks, short weeks: `ball_knower_v1_1.py:65-93`
- Divisional game penalty: -8.3 ELO points (~-0.4 spread points): `ball_knower_v1_1.py:62`
- Surface familiarity: +9.3 ELO points (not fully implemented): `ball_knower_v1_1.py:63`

**Data Range:**
- Week 11, 2025 for predictions
- Adjustment weights derived from nfelo historical pattern analysis (4,510 games, 2009-2025): `ball_knower_v1_1.py:38`

**Evaluation Metrics:**
- Edge vs Vegas: `ball_knower_v1_1.py:212`
- MAE and RMSE of edge: `ball_knower_v1_1.py:283-284`

**What is `bk_line`?**
- `bk_v1_1_spread = base_spread + rest_adj + div_adj + surface_adj`: `ball_knower_v1_1.py:137`
- Intended as football-first prediction with situational context
- Base spread inherited from v1.0 (which was fitted to Vegas)

**Where does Vegas enter?**
- Only as evaluation benchmark: `ball_knower_v1_1.py:168`
- Edge calculated as: `bk_v1_1_spread - spread_line`: `ball_knower_v1_1.py:212`

**Key Finding:**
- Builds on v1.0's Vegas-fitted base, adding theory-driven situational adjustments

---

### ball_knower_v1_2.py (v1.2) ⚠️

**Location:** `/home/user/BK_Build/ball_knower_v1_2.py:1`

**Purpose:** ML correction layer using Ridge regression on historical data

**Target Variable:**
- **`vegas_line = home_line_close`** (Vegas closing spread): `ball_knower_v1_2.py:71`
- **This is the PRIMARY LABEL for training**

**Input Features:**
- `nfelo_diff`: ELO rating differential: `ball_knower_v1_2.py:56`
- `rest_advantage`: Combined home/away bye modifiers: `ball_knower_v1_2.py:61`
- `div_game`: Divisional game indicator: `ball_knower_v1_2.py:63`
- `surface_mod`: Surface change modifier: `ball_knower_v1_2.py:64`
- `time_advantage`: Time zone advantage: `ball_knower_v1_2.py:65`
- `qb_diff`: QB adjustment differential: `ball_knower_v1_2.py:68`

**Data Range:**
- Training: 2009-2024 seasons (4,345 games): `ball_knower_v1_2.py:110`
- Test: 2025 season (165 games): `ball_knower_v1_2.py:111`

**Evaluation Metrics:**
- MAE vs Vegas closing line: `ball_knower_v1_2.py:190-196`
- RMSE vs Vegas closing line
- R² (variance explained in Vegas lines)
- Edge distribution analysis: `ball_knower_v1_2.py:228-232`

**What is `bk_line`?**
- `bk_v1_2_pred = Ridge model prediction`: `ball_knower_v1_2.py:215`
- **This is a direct prediction of what Vegas should be**, not an independent football spread

**Where does Vegas enter?**
- **As the training target** (y variable): `ball_knower_v1_2.py:84`
- As evaluation benchmark for test set: `ball_knower_v1_2.py:216`

**Key Finding:**
- **CRITICAL MISALIGNMENT:** v1.2 is explicitly trained to predict Vegas closing lines, not actual game outcomes
- Model learns: "Given these features, what will Vegas set as the line?"
- This makes "edges" fundamentally a measure of **disagreement with Vegas**, not football insight

---

### bk_v1_final.py (v1.0 variant)

**Location:** `/home/user/BK_Build/bk_v1_final.py:1`

**Purpose:** Another implementation of v1.0 baseline (similar to ball_knower_v1_final.py)

**Target Variable:**
- None (deterministic)

**Input Features:**
- `nfelo_diff` × 0.025 + 2.5 HFA: `bk_v1_final.py:73-75`

**Data Range:**
- Week 11, 2025

**Evaluation Metrics:**
- Absolute edge vs Vegas: `bk_v1_final.py:94`
- RMSE: `bk_v1_final.py:96`

**What is `bk_line`?**
- `bk_spread = -HFA - (nfelo_diff * NFELO_WEIGHT)`: `bk_v1_final.py:75`
- Deterministic spread from power ratings

**Where does Vegas enter?**
- Comparison only: `bk_v1_final.py:76`

**Key Finding:**
- Uses slightly different coefficients (0.025 vs 0.0447) than ball_knower_v1_final.py
- Still compared against Vegas for evaluation

---

### bk_v1_1_with_adjustments.py (v1.1 variant)

**Location:** `/home/user/BK_Build/bk_v1_1_with_adjustments.py:1`

**Purpose:** v1.1 with form adjustments (hot/cold team bonuses)

**Target Variable:**
- None (deterministic)

**Input Features:**
- Base: `nfelo_diff` × 0.025 + 2.5 HFA: `bk_v1_1_with_adjustments.py:60-63`
- Recent form (L5 record): ±0.5 points for hot/cold teams: `bk_v1_1_with_adjustments.py:68-98`
- Rest advantage (placeholder): `bk_v1_1_with_adjustments.py:102`

**Data Range:**
- Week 11, 2025

**Evaluation Metrics:**
- Edge vs Vegas: `bk_v1_1_with_adjustments.py:108-109`
- MAE and RMSE: `bk_v1_1_with_adjustments.py:135-142`

**What is `bk_line`?**
- `adjusted_spread = base_spread + form_adj + rest_adj`: `bk_v1_1_with_adjustments.py:105`

**Where does Vegas enter?**
- Comparison benchmark: `bk_v1_1_with_adjustments.py:168`

**Key Finding:**
- Adds form-based adjustments to deterministic base

---

## Backtest Scripts

### backtest_v1_0.py

**Location:** `/home/user/BK_Build/backtest_v1_0.py:1`

**Purpose:** Historical backtest of v1.0 model against Vegas lines

**Data Range:**
- 2009-2025 (4,510 games with complete data): `backtest_v1_0.py:54`

**What it measures:**
- Prediction accuracy **vs Vegas closing lines**, NOT actual game outcomes: `backtest_v1_0.py:69`
- Edge distribution by threshold: `backtest_v1_0.py:169-188`
- **NO ACTUAL GAME RESULTS** - only measures agreement with Vegas: `backtest_v1_0.py:195-207`

**Key Metrics:**
- MAE vs Vegas: `backtest_v1_0.py:85`
- RMSE vs Vegas: `backtest_v1_0.py:86`
- R² (variance in Vegas lines explained): `backtest_v1_0.py:87`

**Critical Limitation (acknowledged in script):**
- "NO ACTUAL RESULTS: This backtest only measures edge vs Vegas, not actual game outcomes. To calculate real ROI, we'd need to know which side covered.": `backtest_v1_0.py:199-201`
- "LOOK-AHEAD BIAS: This model was calibrated on 2025 Week 11 data and applied to historical data.": `backtest_v1_0.py:195-197`

**Key Finding:**
- Script explicitly acknowledges it's NOT measuring betting performance, only Vegas agreement

---

### backtest_v1_2.py ⚠️

**Location:** `/home/user/BK_Build/backtest_v1_2.py:1`

**Purpose:** Professional backtest with EV, Kelly criterion, and ROI simulation

**Data Range:**
- Training: 2009-2024
- Test: 2025 season

**What it measures:**
- Prediction accuracy vs Vegas: `backtest_v1_2.py:189-195`
- Expected Value (EV) calculations assuming -110 odds: `backtest_v1_2.py:147-172`
- Kelly criterion bet sizing: `backtest_v1_2.py:161-170`
- **Still no actual game outcomes used**: `backtest_v1_2.py:411-417`

**Key Metrics:**
- MAE and RMSE vs Vegas
- Hypothetical EV per bet: `backtest_v1_2.py:213-231`
- Kelly sizing recommendations: `backtest_v1_2.py:233-250`

**Critical Limitation (acknowledged in script):**
- "No actual game outcomes used - theoretical edge only": `backtest_v1_2.py:412`
- "These EV calculations assume -110 odds on all bets (simplified)": `backtest_v1_2.py:411`

**Key Finding:**
- Despite sophisticated betting analytics (EV, Kelly), it still measures "edge vs Vegas" not "profitability vs actual outcomes"
- The EV calculations are **theoretical** based on model's probability estimates, not validated against real game results

---

## Calibration Scripts

All four calibration scripts share a common objective: **fit model weights to minimize error vs Vegas lines**.

### calibrate_model.py

**Location:** `/home/user/BK_Build/calibrate_model.py:1`

**Target:** Match Vegas spread lines: `calibrate_model.py:100-113`

**Method:** Optimize EPA weight to minimize MSE vs Vegas

**Data:** 2015-2024 historical games with EPA and Vegas lines

**Key Finding:** Explicitly fits to Vegas as target

---

### calibrate_regression.py

**Location:** `/home/user/BK_Build/calibrate_regression.py:1`

**Target:** `spread_line` (Vegas closing line): `calibrate_regression.py:70`

**Method:** Linear regression with Vegas line as y variable

**Data:** Week 11, 2025 (14 games)

**Output:** Coefficients that minimize error vs Vegas: `calibrate_regression.py:74-90`

**Key Finding:** All three models (nfelo only, nfelo+EPA, nfelo+EPA+substack) are fitted to predict Vegas

---

### calibrate_simple.py

**Location:** `/home/user/BK_Build/calibrate_simple.py:1`

**Target:** `spread_line` (Vegas closing line): `calibrate_simple.py:35`

**Method:** Scalar optimization to minimize MSE vs Vegas: `calibrate_simple.py:62-67`

**Data:** Week 11, 2025

**Key Finding:** Fits nfelo weight specifically to match Vegas

---

### calibrate_to_vegas.py

**Location:** `/home/user/BK_Build/calibrate_to_vegas.py:1`

**Target:** Current Vegas lines: `calibrate_to_vegas.py:26`

**Method:** Multi-parameter optimization vs Vegas: `calibrate_to_vegas.py:91-96`

**Quote:** "Simpler approach: Fit model weights to minimize difference from current Vegas lines. This aligns with the principle: 'Model the market bias, not the game.'": `calibrate_to_vegas.py:6-7`

**Key Finding:**
- **EXPLICIT DESIGN CHOICE** to model Vegas, not football
- Script literally says the goal is to "model the market bias, not the game"

---

## Summary Table

| Script | Purpose | Target Variable | Vegas Role | Football Target? |
|--------|---------|----------------|------------|------------------|
| **ball_knower_v1_final.py** | v1.0 baseline | None (deterministic) | Comparison + coefficient calibration | Intended, but coefficients fitted to Vegas |
| **ball_knower_v1_1.py** | v1.1 situational | None (deterministic) | Comparison only | Yes, adjustments theory-driven |
| **ball_knower_v1_2.py** | v1.2 ML layer | **vegas_line** | **Training target** | ❌ **NO - predicts Vegas** |
| **bk_v1_final.py** | v1.0 variant | None (deterministic) | Comparison | Intended |
| **bk_v1_1_with_adjustments.py** | v1.1 variant | None (deterministic) | Comparison | Yes |
| **backtest_v1_0.py** | v1.0 backtest | N/A | Evaluation benchmark | ❌ No actual outcomes used |
| **backtest_v1_2.py** | v1.2 backtest | N/A | Evaluation benchmark | ❌ No actual outcomes used |
| **calibrate_model.py** | Fit EPA weight | Vegas spread | **Optimization target** | ❌ NO |
| **calibrate_regression.py** | Fit linear model | Vegas spread | **Optimization target** | ❌ NO |
| **calibrate_simple.py** | Fit nfelo weight | Vegas spread | **Optimization target** | ❌ NO |
| **calibrate_to_vegas.py** | Fit all weights | Vegas spread | **Optimization target** | ❌ NO - explicit design |

---

## Objective Misalignment and Risks

### Where are we clearly predicting Vegas instead of football?

1. **ball_knower_v1_2.py (PRIMARY ISSUE)**
   - Training target is `vegas_line = home_line_close`: `ball_knower_v1_2.py:71`
   - Model learns: "What spread will Vegas set?" not "What spread should reflect the true game?"
   - This is **regression to the Vegas mean**, not independent football analysis

2. **All calibration scripts**
   - `calibrate_model.py` minimizes MSE vs Vegas spreads: `calibrate_model.py:100-113`
   - `calibrate_regression.py` uses Vegas spread as y variable: `calibrate_regression.py:70`
   - `calibrate_simple.py` optimizes to match Vegas: `calibrate_simple.py:62-67`
   - `calibrate_to_vegas.py` **explicitly states** goal is to "model the market bias, not the game": `calibrate_to_vegas.py:6-7`

3. **v1.0 deterministic models (INDIRECT)**
   - While conceptually "theory-driven," the coefficients (0.0447, 2.67) were derived by fitting to Vegas lines
   - This means even the "independent" baseline is anchored to Vegas

### Why is this risky?

**When the model diverges from Vegas, it's often because the model is wrong, not Vegas**
- Vegas closing lines incorporate:
  - Sharp money
  - Injury news
  - Weather updates
  - Line shopping across books
  - Decades of market efficiency learning

- Our model uses:
  - Historical ELO ratings
  - Generic situational factors
  - No real-time information
  - Limited feature set

**We are rewarding the model for matching Vegas, so big differences tend to be errors**
- During training (v1.2), the model is penalized for deviating from Vegas
- This creates a self-fulfilling prophecy: "edges" are by definition model errors relative to the market
- Larger edges = larger model errors = worse expected performance

**This makes "edges" vs Vegas mostly a measure of model error**
- If we're predicting Vegas and Vegas is the ground truth, then:
  - `edge = our_prediction - vegas = model_error`
- When we have a "big edge," it means: "our model strongly disagrees with the efficient market"
- In efficient markets, strong disagreement usually means **we're wrong**, not that we found alpha

### What has already been observed in prior experiments?

**Previous analysis finding (from earlier v2.0 review):**

In a prior session's high-level review, the following pattern was identified:

> "Large 'edges' vs Vegas often performed worse than small edges, which is exactly backwards from what we'd expect if we had genuine football insight."

**Approximate empirical results (from previous analysis):**

| Edge Bin | Expected Behavior | Observed Behavior |
|----------|-------------------|-------------------|
| 0-1 point | Small edge, modest value | Better ATS performance |
| 1-2 points | Medium edge, good value | Moderate ATS performance |
| 2-3 points | Large edge, strong value | **Worse ATS performance** |
| 3+ points | Huge edge, max value | **Worst ATS performance** |

**Interpretation:**
- If our model had genuine football insight independent of Vegas, we'd expect:
  - Larger edges → better performance (we see something Vegas missed)
- Instead we observe:
  - Larger edges → worse performance (we're wrong and Vegas is right)

This is the **signature pattern of a model that's trying to predict Vegas but adding noise** rather than providing independent football analysis.

### Root Cause

The fundamental issue traces back to a design philosophy visible in `calibrate_to_vegas.py:6-7`:

> "Simpler approach: Fit model weights to minimize difference from current Vegas lines. This aligns with the principle: 'Model the market bias, not the game.'"

**This principle is backwards for betting.**

**Correct principle:** Build a model of the game, then find spots where the market is biased.

**Current approach:** Build a model of the market, then be surprised when deviations from the market perform poorly.

### Implications for the codebase

1. **v1.0 and v1.1 models** are partially salvageable
   - The *structure* (nfelo + situational factors) is sound
   - But coefficients need to be re-derived from actual game outcomes, not Vegas lines

2. **v1.2 ML layer** requires fundamental redesign
   - Cannot use `vegas_line` as training target
   - Should predict actual margin, cover probability, or other football outcomes
   - Vegas can be an input feature, but not the label

3. **Backtests** need actual game results
   - Current backtests only measure "how well did we match Vegas?"
   - Need to measure "how profitable were our bets when games were played?"
   - Requires joining predictions with actual scores and calculating ATS results

4. **Calibration approach** needs inversion
   - Stop minimizing error vs Vegas
   - Start minimizing error vs actual game margins (or maximizing ATS win rate on holdout set)
   - Use Vegas information as a feature or reality check, not the optimization target

---

## Recommendations (Documentation Phase Only)

**DO NOT IMPLEMENT THESE YET** - this is documentation/audit only.

For future modeling phases:

1. **Establish a clear target hierarchy:**
   - Primary: Actual game margin or cover probability
   - Secondary: Implied win probability
   - Tertiary: Vegas comparison (for calibration checks only)

2. **Rebuild v1.0 coefficients:**
   - Fit nfelo/EPA weights to actual game margins, not Vegas lines
   - This gives us a true "football baseline"

3. **Redesign v1.2 training:**
   - Change target from `vegas_line` to `actual_margin` or `covered` (binary)
   - Add `vegas_line` as an input feature (it contains information)
   - Evaluate on ATS performance on holdout set

4. **Fix backtesting:**
   - Join predictions with actual game outcomes
   - Calculate: "If we bet every game with edge >= X, what was our ATS record?"
   - Track ROI in units, not just "agreement with Vegas"

5. **Document feature tiers:**
   - Create/update FEATURE_TIERS.md to clarify which features belong in which model version
   - Ensure Tier 1 (stable, structural) stays in v1.0/v1.1
   - Keep advanced/volatile stats for v1.2+

---

**End of Model Overview**

For forward-looking design recommendations, see `BALL_KNOWER_MODELING_PLAN.md`.
