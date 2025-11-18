# Ball Knower Modeling Plan

**Version:** 2.0 (Post-Audit)
**Date:** 2025-11-18
**Status:** Design Document (Not Yet Implemented)

---

## Objective

**Define the core, long-term goal in plain language:**

Build a **football-first point spread prediction system** that:

1. **Produces an independent spread (`bk_line`)** based on pre-game football fundamentals (power ratings, situational factors, structural features)

2. **Derives probability distributions** over game margins from our spread estimates

3. **Identifies betting value** where our football-based probabilities meaningfully diverge from market prices for sound, explainable reasons

4. **Prioritizes calibration and robustness** over fitting to historical Vegas lines or overfitting to limited data

**Core Principle:**
Model the **game**, not the **market**. Use market information as context and validation, but never as the primary training target.

---

## Versioned Roadmap (Conceptual)

This section describes what each version should accomplish. No code changes required in this phase.

### v1.0 – Deterministic Base Spread (Theory-Driven)

**Goal:** Establish a stable, interpretable baseline using only power ratings and theory

**Approach:**
- Purely deterministic function of nfelo (or similar power rating) + home field advantage
- Coefficients should be derived from **actual game margin analysis**, not from fitting to Vegas
- Example: "40 nfelo points ≈ 1 spread point" based on historical game score differentials
- NO machine learning, NO optimization to Vegas

**Output:**
- `bk_line_v1_0`: Our raw football rating-based spread
- Treat this as the "null hypothesis" - what the spread should be based purely on team quality

**Validation:**
- Evaluate against actual game margins (MAE, RMSE vs actual margin)
- Secondarily check correlation with Vegas (should be high if both model football well)
- Do NOT optimize coefficients to match Vegas

**Success Criteria:**
- RMSE vs actual margin ≈ 13-14 points (inherent NFL variance)
- R² vs actual margin ≈ 0.20-0.30 (power ratings alone explain some but not all variance)
- Clear documentation of coefficient derivation methodology

---

### v1.1 – Structural Calibration (Cheap, Stable Features)

**Goal:** Refine v1.0 with low-variance structural adjustments

**Approach:**
- Start with v1.0 base spread
- Add **cheap, stable, structural** features only:
  - Home/away
  - Dome vs outdoor
  - Altitude
  - Rest days / bye weeks
  - Short week flags
  - Divisional game flags
  - Travel distance (optional)
- These features should be:
  - **Available pre-game** (no leakage)
  - **Low noise** (not subject to weekly volatility)
  - **Theory-backed** (clear causal story for why they matter)

**Output:**
- `bk_line_v1_1 = bk_line_v1_0 + structural_adjustments`
- Adjustments should be small (total ±2-3 points max)

**Validation:**
- Same as v1.0: evaluate vs actual game margins first
- Check if MAE vs actual margin improves (should drop slightly)
- Validate that adjustments make football sense (e.g., bye week team performs better on average)

**Success Criteria:**
- RMSE vs actual margin < v1.0 (even if only marginally)
- Adjustments are explainable and directionally correct
- No overfitting - use simple, fixed weights or small regularized fits

---

### v1.2 – ML Spread Correction (Football Target, Regularized)

**Goal:** Learn small corrections to v1.0/v1.1 spread using historical game data

**CRITICAL CHANGE FROM CURRENT v1.2:**
- **Target variable MUST be football-based**, NOT `vegas_line`
- Recommended targets (choose one):
  1. **Actual game margin** (`home_score - away_score`)
  2. **Cover outcome** (binary: did favorite cover?)
  3. **Residual from v1.1** (what did v1.1 miss?)

**Approach:**
- Use v1.1 spread as a feature (or base prediction)
- Add additional features (EPA, recent form, advanced stats) - **Tier 2 from FEATURE_TIERS**
- Train a **small, regularized model** (Ridge, Lasso, or LightGBM with strong regularization)
- Vegas spread **can be an input feature** (it contains information), but NOT the target
- Use proper time-series cross-validation

**Output:**
- `bk_line_v1_2`: Refined football spread that corrects v1.1's systematic biases
- Model learns: "When v1.1 says X, the actual game tends to be Y"
- NOT: "When v1.1 says X, Vegas tends to set Y"

**Validation:**
- **Primary:** MAE/RMSE vs actual game margins on holdout set
- **Secondary:** ATS performance (if we bet every game, what % would we cover?)
- **Tertiary:** Compare to Vegas as a sanity check (should correlate but not perfectly)

**Success Criteria:**
- Marginal improvement in RMSE vs actual margins compared to v1.1
- No signs of overfitting (train vs test performance gap < 5%)
- Feature weights are interpretable and directionally sensible

---

### v2.0 – Betting Decision Layer (Meta-Model)

**Goal:** Convert football spreads into betting decisions with bankroll management

**Approach:**
- Take `bk_line_v1_2` (our football spread) and `market_line` (current Vegas spread)
- Build a **meta-model** that predicts:
  - **Probability of covering** the market spread (not our spread)
  - **Expected value** of betting each side at current odds
  - **Recommended bet size** using Kelly criterion or similar
- Incorporate additional betting-specific signals:
  - Line movement (steam, reverse line movement)
  - Public betting percentages (contrarian opportunities)
  - Sharp vs public money indicators
  - Time until kickoff (line value decay)

**Output:**
- For each game:
  - `bet_recommendation`: {None, Home, Away}
  - `bet_size`: Fraction of bankroll (0-5%)
  - `expected_value`: EV in dollars per $100 staked
  - `confidence_level`: Low/Medium/High

**Validation:**
- **Primary:** ROI on actual bets placed in holdout period
- **Secondary:** Sharpe ratio, max drawdown, Kelly compliance
- **Tertiary:** ATS win rate stratified by confidence level

**Success Criteria:**
- Positive ROI on holdout set (even if small)
- High-confidence bets outperform low-confidence bets
- Bankroll management prevents ruin scenarios

---

## Principles / Guardrails

**Core Modeling Principles** (to be followed in all future work):

1. **Never train a model to directly predict the Vegas closing spread as the primary objective**
   - Vegas spread can be a feature, benchmark, or reality check
   - It cannot be the y variable in supervised learning

2. **All targets must be defined in terms of actual game outcomes**
   - Preferred: actual margin, actual score, cover outcome (binary), win probability
   - Avoid: residuals from Vegas, "what should Vegas have set"

3. **Vegas information can be used as an input feature or benchmark, not the primary label**
   - It's valid to include `vegas_line` as a feature (it contains real information)
   - It's invalid to optimize model weights to minimize `MAE(prediction, vegas_line)`

4. **Use time-aware splits and never leak future information into training**
   - Always split train/test by time (e.g., train on 2009-2023, test on 2024)
   - Use `.shift(1)` or equivalent for rolling averages
   - No same-week or forward-looking features

5. **Keep feature sets simple and explainable; start small, add complexity only when justified**
   - Prefer 5-10 strong features over 50 weak features
   - Each feature should have a clear football rationale
   - Complexity budget: only add features that improve holdout performance

6. **Regularization is mandatory for ML models**
   - Always use Ridge, Lasso, Elastic Net, or tree-based regularization
   - Prevents overfitting to noise in small sample sizes (16-17 games/team/season)
   - Tune regularization strength via cross-validation

7. **Backtest on actual game results, not Vegas agreement**
   - Primary metric: ROI if we had bet based on model recommendations
   - Secondary: ATS win rate, cover probability calibration
   - Tertiary: Correlation with Vegas (as a sanity check only)

8. **Document all coefficient sources and derivations**
   - If a weight comes from research, cite it
   - If derived from data, document the dataset and methodology
   - Never use "magic numbers" without justification

9. **Treat Vegas as a competitor, not ground truth**
   - Vegas is very good but not perfect
   - Our goal: find the 5-10% of games per week where Vegas is systematically wrong
   - If we're always close to Vegas, we have no edge

10. **Prioritize robustness over backtested performance**
    - A model with 52% ATS win rate that's stable across 5 seasons beats a model with 60% that only worked in 2023
    - Avoid overfitting to any one season, regime, or rule change

---

## Data & Feature Tiers

This section references the existing feature engineering strategy and maps it to model versions.

### Tier 1: Stable, Structural Features (v1.0, v1.1)

**Characteristics:**
- Available pre-game with no leakage risk
- Low variance, not sensitive to weekly noise
- Clear causal mechanism

**Examples:**
- Power ratings (nfelo, EPA season-to-date)
- Home field advantage
- Rest days, bye weeks
- Dome vs outdoor
- Altitude
- Divisional game flag
- Travel distance

**Model versions:**
- v1.0 uses: power ratings + HFA only
- v1.1 adds: rest, dome, divisional flags

---

### Tier 2: Informative but Noisier Features (v1.2)

**Characteristics:**
- More predictive but higher variance
- Require careful feature engineering to avoid leakage
- May need regularization to prevent overfitting

**Examples:**
- Recent form (L3, L5 record)
- Rolling EPA (properly lagged)
- Strength of schedule
- QB performance metrics (season-to-date)
- Offensive/defensive tendency stats
- Weather conditions (if available pre-game)

**Model version:**
- v1.2 adds these on top of v1.1 base

---

### Tier 3: Advanced / Volatile Features (v2.0+)

**Characteristics:**
- Potentially high value but high risk of overfitting
- Require large datasets and sophisticated validation
- May not generalize across seasons

**Examples:**
- Play-calling tendencies vs specific defenses
- Advanced coaching analytics
- Injury impact estimates
- Sentiment / public bias signals
- Line movement features
- Sharp vs public money indicators

**Model version:**
- v2.0 betting layer can experiment with these carefully
- Should be validated on multi-year holdout sets

---

### Reference: FEATURE_TIERS.md

If a `FEATURE_TIERS.md` document exists or will be created, it should:
- Expand on the above categorization
- Provide detailed definitions of each feature
- Document leakage risks and mitigation strategies
- Show example code for proper feature engineering
- Include validation tests for each feature

---

## Transition Plan: From Current State to New Architecture

**Current state:**
- v1.0/v1.1: Deterministic, coefficients fitted to Vegas
- v1.2: ML model trained to predict Vegas closing line
- Backtests: Measure agreement with Vegas, not betting profitability

**Target state:**
- v1.0/v1.1: Deterministic, coefficients derived from actual game margins
- v1.2: ML model trained to predict actual game outcomes
- Backtests: Measure ATS win rate and ROI on actual bet simulations

**Transition steps (for future implementation phase):**

1. **Create v1.0 "refit":**
   - Load historical game data (2009-2024) with nfelo ratings and actual scores
   - Calculate actual margins: `actual_margin = home_score - away_score`
   - Fit: `actual_margin ~ intercept + nfelo_diff`
   - Use these new coefficients as v1.0 baseline
   - Document change in methodology

2. **Validate v1.0 refit:**
   - Does it still correlate well with Vegas? (Should, if both model football)
   - Does RMSE vs actual margin improve?
   - Are "edges" now more meaningful?

3. **Rebuild v1.2 with new target:**
   - Change: `y = df['vegas_line']` → `y = df['actual_margin']`
   - Retrain Ridge model
   - Validate on actual game outcomes in 2024/2025

4. **Build backtest harness:**
   - Join model predictions with actual game results
   - Simulate: "If we bet games with edge >= X, what was our ATS record?"
   - Calculate ROI, Sharpe ratio, max drawdown

5. **Compare old vs new:**
   - Does new football-first approach have better ATS performance?
   - Are larger edges now more profitable (as they should be)?
   - Document findings

---

## Success Metrics by Model Version

### v1.0 Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| RMSE vs actual margin | 13-14 points | Inherent NFL variance baseline |
| R² vs actual margin | 0.20-0.30 | Power ratings explain some variance |
| Correlation with Vegas | 0.80-0.90 | Both model same game, should agree often |
| MAE vs actual margin | 10-11 points | More interpretable error measure |

### v1.1 Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| RMSE vs actual margin | < v1.0 by 0.5-1 point | Small improvement from structural features |
| R² vs actual margin | 0.25-0.35 | Modest increase |
| Adjustment magnitude | ±2-3 points max | Structural features are small effects |
| Directional accuracy | > 70% | Adjustments should help more than hurt |

### v1.2 Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| RMSE vs actual margin | < v1.1 by 0.5-1 point | ML should capture nonlinear patterns |
| R² vs actual margin | 0.30-0.40 | Diminishing returns expected |
| Train/test gap | < 5% | Regularization prevents overfitting |
| ATS win rate (all bets) | 51-53% | Slight edge if model is well-calibrated |

### v2.0 Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| ROI (all recommended bets) | > 0% | Profitability after vig |
| ROI (high confidence only) | > 3% | Strong bets should be clearly +EV |
| ATS win rate (high confidence) | > 55% | Selective betting should outperform |
| Max drawdown | < 20 units | Bankroll management works |
| Sharpe ratio | > 0.5 | Risk-adjusted returns competitive |

---

## Risk Factors and Mitigations

### Risk 1: NFL is a Small-Sample Environment

**Challenge:**
- Only 16-17 games per team per season
- Only ~270 games per season league-wide
- Models can easily overfit

**Mitigation:**
- Strong regularization in all ML models
- Prefer simple models (linear, shallow trees) over complex
- Multi-year cross-validation required
- Conservative out-of-sample testing

---

### Risk 2: Market Efficiency

**Challenge:**
- NFL betting markets are highly efficient
- Sharp bettors move lines quickly
- Closing line value (CLV) is hard to beat

**Mitigation:**
- Don't try to beat sharp bettors on every game
- Focus on systematic biases (e.g., public overreaction to narratives)
- Be selective (5-10 bets/week, not 15)
- Track CLV but don't obsess over it

---

### Risk 3: Data Leakage in Features

**Challenge:**
- Easy to accidentally use future information
- Rolling stats can leak if not properly shifted
- Vegas lines themselves can leak (if we fit to them)

**Mitigation:**
- Mandatory `.shift(1)` for all rolling features
- Strict time-based train/test splits
- Code reviews for feature engineering
- Never optimize to Vegas as a target

---

### Risk 4: Regression to the Mean

**Challenge:**
- Teams that outperform early in season regress
- Hot/cold streaks are often noise, not signal
- Models trained on recent data may chase variance

**Mitigation:**
- Use full-season power ratings, not recent form, as base
- Limit weight on L3/L5 record
- Validate that "form" features actually predict future performance on holdout

---

### Risk 5: Changing NFL Dynamics

**Challenge:**
- Rule changes affect scoring
- Coaching/play-calling evolves
- Models trained on 2015 data may not work in 2025

**Mitigation:**
- Weighted recency in training (recent years matter more)
- Re-train models annually
- Monitor performance degradation year-over-year
- Be ready to adapt feature sets

---

## Open Questions for Future Phases

These questions should be addressed during implementation, not in this doc phase:

1. **Should v1.0 use nfelo, EPA, or a combination?**
   - Test both on actual game margins
   - May depend on data availability

2. **What's the optimal regularization strength for v1.2?**
   - Requires cross-validation experiments
   - Likely alpha ∈ [1, 100] for Ridge

3. **Should v1.2 predict margin or cover probability?**
   - Margin: more information, harder to calibrate
   - Cover probability: directly actionable, simpler
   - Could build both and compare

4. **How many seasons of data for training?**
   - Trade-off: more data (better stats) vs recency (NFL changes)
   - Likely 5-10 years is optimal

5. **What edge threshold for betting?**
   - Needs empirical testing on holdout set
   - Likely 1.5-3 points depending on confidence

6. **How to handle missing features (injuries, weather)?**
   - Imputation strategy?
   - Separate "weather adjustment" layer?

---

## Next Steps (For Future Implementation)

**This document is a design spec, not an implementation guide.**

When ready to begin implementation:

1. **Phase 1: Rebuild v1.0**
   - Fit nfelo coefficients to actual game margins
   - Validate on 2024/2025 holdout
   - Document methodology change

2. **Phase 2: Enhance to v1.1**
   - Add structural features with theory-driven weights
   - Validate improvement on actual margins

3. **Phase 3: Retrain v1.2**
   - Change target to actual margin
   - Add Tier 2 features
   - Cross-validate and test

4. **Phase 4: Build Backtest Harness**
   - Join predictions + actual results
   - Calculate ATS win rate by edge bucket
   - Validate if larger edges now perform better

5. **Phase 5: Implement v2.0 Betting Layer**
   - Build meta-model for bet sizing
   - Add line movement features
   - Real-money pilot testing (small stakes)

---

## Appendix: Comparison of Old vs New Philosophy

| Aspect | Old Approach (Current State) | New Approach (This Plan) |
|--------|------------------------------|--------------------------|
| **Primary Goal** | Match Vegas lines closely | Model football outcomes independently |
| **Training Target** | `vegas_line` (closing spread) | `actual_margin` or `cover_outcome` |
| **Vegas Role** | Optimization target | Input feature + benchmark |
| **Edge Definition** | Difference from Vegas | Probability-weighted disagreement |
| **Success Metric** | MAE vs Vegas | ATS win rate, ROI |
| **Large Edges** | Often model errors | Potentially valuable bets (if model is good) |
| **Philosophy** | "Model the market" | "Model the game, find market inefficiencies" |

---

**End of Modeling Plan**

This document should be treated as the **design contract** for all future Ball Knower modeling work.

Before implementing any model changes, validate that the approach aligns with the principles and roadmap defined here.
