# STEP 2 Assessment: EPA Data Access Challenges

## Current Situation

**Network Restrictions Blocking EPA Data:**
- Both `nfl_data_py` (deprecated) and `nflreadpy` (modern) return 403 Forbidden errors
- Cannot access NFLverse play-by-play data from GitHub releases
- Cannot download EPA, success rate, or advanced efficiency metrics
- This blocks the originally planned STEP 2 (Modernize Features with EPA/DVOA)

**Available Data:**
- ✓ greerreNFL/nfelo historical ratings (2009-2025)
- ✓ Current week schedule data from nflverse API
- ✓ Vegas lines (opening and closing)
- ✓ Situational factors (rest, travel, QB, etc.)

**Not Available:**
- ✗ Play-by-play data for EPA calculation
- ✗ Pre-computed team efficiency metrics
- ✗ Weather data
- ✗ Game outcomes/scores (for training)

## Three Options Forward

### OPTION A: Feature Engineering with Current Data
**Approach:** Enhance v1.2 with derived features from nfelo
- Rolling averages of ELO changes (momentum)
- Interaction terms (rest × ELO diff)
- Polynomial features for nonlinearity
- Team-specific HFA adjustments

**Pros:**
- No dependency on external data
- Can implement immediately
- Low risk

**Cons:**
- Limited improvement potential (5-10% MAE reduction)
- Still predicting spread corrections (not true outcomes)
- Doesn't address core architectural limitation

**Estimated effort:** 2-3 hours
**Expected improvement:** MAE from 1.57 → 1.45 points

---

### OPTION B: Manual EPA Data Integration
**Approach:** Download EPA data manually, store locally
- Find alternative EPA data source (CSV format)
- Download and commit to repo
- Build EPA aggregation pipeline
- Retrain with EPA features

**Pros:**
- Gets us the "professional" features from research
- Significant performance improvement potential
- Aligns with original STEP 2 plan

**Cons:**
- Requires manual data sourcing
- May hit same network restrictions
- Adds complexity to pipeline
- Still predicting spreads (not outcomes)

**Estimated effort:** 4-6 hours
**Expected improvement:** MAE from 1.57 → 1.25 points

---

### OPTION C: Skip to STEP 3 (Score Prediction Model) ⭐ RECOMMENDED
**Approach:** Build unified score-per-team model with current features

**Architecture:**
```
For each game:
  Input: [nfelo_diff, rest_diff, qb_diff, div_game, situational_factors]

  → Model A: Predict home_score
  → Model B: Predict away_score

  Output: (home_score, away_score) distribution

  Derive:
  - Spread = home_score - away_score
  - Total = home_score + away_score
  - Moneyline odds from win probability
  - Full game simulation capabilities
```

**Implementation Options:**
1. **Dual Ridge Regression** (simple, fast)
   - Separate models for home/away scores
   - Same features as v1.2
   - Interpretable coefficients

2. **Poisson/Negative Binomial** (principled)
   - Natural fit for score distributions
   - Handles discrete outcomes properly
   - Industry standard for soccer/hockey

3. **Gradient Boosting** (powerful)
   - XGBoost or LightGBM
   - Captures nonlinear interactions
   - Better calibration for probabilities

**Pros:**
- Unlocks moneyline and totals betting
- True win probabilities (not spread-derived)
- Full probability distributions
- Can simulate game outcomes
- Bigger gains from architecture than features
- Works with current data sources

**Cons:**
- More complex than spread correction model
- Need to validate score predictions make sense
- May have lower accuracy on individual scores vs spreads

**Estimated effort:** 3-4 hours
**Expected improvement:**
- Spread MAE: Similar to v1.2 (1.55-1.65 points)
- NEW: Moneyline accuracy, totals prediction
- NEW: Full probability distributions

## Recommendation: OPTION C

### Why Skip STEP 2 and Go to STEP 3?

1. **Network constraints are blocking EPA data access**
   - Can't get play-by-play data in this environment
   - Manual workarounds add complexity without guarantees

2. **Architectural improvement > Feature improvement**
   - Score prediction unlocks new bet types
   - Aligns better with professional research (outcome-based, not Vegas-based)
   - More valuable for actual betting applications

3. **Current features are already strong**
   - v1.2 MAE of 1.57 points is excellent
   - nfelo ratings are well-regarded in the industry
   - Marginal gains from EPA may not justify effort

4. **Can revisit EPA features later**
   - Once network access improves, can add EPA to score model
   - Score model provides better foundation for features
   - v1.3 → v1.4 path with EPA still available

### Revised Roadmap

✅ **STEP 1 (COMPLETE):** Professional betting framework
- betting_utils.py with CLV/EV/Kelly
- Comprehensive backtesting
- Win probability calculations

❌ **STEP 2 (DEFERRED):** EPA/DVOA features
- Blocked by network restrictions
- Can revisit when data access improves

⏭️ **STEP 3 (NEXT):** Unified score prediction model
- Build dual-score architecture
- Train on nfelo + situational features
- Derive spreads, totals, moneylines
- Compare to v1.2 baseline

🔮 **STEP 4 (FUTURE):** Enhanced features when possible
- Add EPA when data accessible
- Weather integration
- Player props
- Live betting models

## Next Actions

If proceeding with OPTION C (STEP 3):

1. Design score model architecture
   - Choose: Dual Ridge vs Poisson vs GBM
   - Define training target (game scores)
   - Handle score correlation (home/away are related)

2. Get game outcome data
   - Check if nfelo has results
   - Or use nflverse current week results for validation
   - Build historical score database

3. Train and validate
   - Same train/test split as v1.2 (2025 holdout)
   - Compare derived spreads to v1.2 predictions
   - Evaluate moneyline and totals accuracy

4. Integrate with betting framework
   - Calculate true win probabilities from score distributions
   - Derive moneyline odds
   - Calculate totals probabilities (over/under)
   - Update prediction script with all bet types

---

**Decision Point:** Proceed with OPTION C (score prediction)?
