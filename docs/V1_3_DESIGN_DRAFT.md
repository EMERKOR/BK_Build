# Ball Knower v1.3: Meta-Edge & Market Intelligence Layer

**Status:** 📋 **DESIGN DRAFT** (Not Yet Implemented)
**Target Release:** Q1 2026 (After v1.2 backtest validation)
**Design Date:** 2025-11-17
**Author:** Claude (claude-sonnet-4-5)

---

## Purpose & Vision

Ball Knower v1.3 introduces a **meta-edge and market intelligence layer** that learns from betting market dynamics, public sentiment, and situational factors beyond pure team performance metrics.

### Conceptual Evolution

```
v1.0: Pure Team Performance
├─ EPA margins, power ratings, HFA
└─ "Which team is better?"

v1.2: Residual Correction on Team Performance
├─ Learns what v1.0 systematically misses
└─ "How wrong is our team-based model?"

v1.3: Market Intelligence & Meta-Edge (PROPOSED)
├─ Line movement, public betting, situational spots
└─ "What does the MARKET know that we don't?"
```

### Key Hypothesis

**The betting market incorporates information not captured by team performance metrics alone:**

1. **Line Movement** - Sharp action moves lines before game time
2. **Public Bias** - Certain teams/situations attract lopsided public betting
3. **Situational Spots** - Lookahead games, letdown spots, revenge games
4. **Injury Impact** - Late-breaking injury news affects spreads
5. **Weather/Environment** - Extreme conditions favor certain styles
6. **Coaching Edges** - Certain coaches outperform in specific matchups

**v1.3 Goal:** Learn which market signals are predictive and which are noise.

---

## Model Architecture (Proposed)

### Option A: Third-Layer Residual Model (Recommended)

```
Input: Canonical team features + market features
   │
   ├──► v1.0 Base Model (Deterministic)
   │        └──► base_prediction
   │
   ├──► v1.2 Correction Model (ML Residual 1)
   │        └──► correction_1 = learn(Vegas - base_prediction)
   │        └──► v1_2_prediction = base_prediction + correction_1
   │
   └──► v1.3 Market Meta-Model (ML Residual 2)
            └──► correction_2 = learn(Vegas - v1_2_prediction, market_features)
            └──► v1_3_prediction = v1_2_prediction + correction_2
```

**Advantages:**
- ✅ Preserves v1.0 deterministic logic
- ✅ Preserves v1.2 learned corrections
- ✅ v1.3 only learns from **market signals**, not team performance
- ✅ Modular: Can disable v1.3 layer if market data unavailable
- ✅ Clear separation: Team performance → Performance corrections → Market intelligence

**Training:**
- v1.3 learns: `residual_2 = vegas_line - v1_2_prediction`
- Features: **ONLY** market/situational features (no team performance)
- Target: What's left after v1.2 has already corrected team-based errors

### Option B: Ensemble Meta-Model

```
Input: Canonical features + market features
   │
   ├──► v1.0 Base Model → prediction_v1_0
   ├──► v1.2 Correction Model → prediction_v1_2
   └──► v1.3 Meta-Model (Ensemble)
            └──► Combines v1_0, v1_2, market features
            └──► Weighted combination or stacking model
```

**Advantages:**
- ✅ Can learn optimal weights for different model contributions
- ✅ Can down-weight v1.0/v1.2 in situations where market knows better

**Disadvantages:**
- ❌ More complex architecture
- ❌ Harder to interpret what v1.3 is learning
- ❌ Risk of overfitting to v1.0/v1.2 outputs

**Recommendation:** Start with **Option A** (third-layer residual) for simplicity and interpretability.

---

## Feature Design: Market Intelligence Features

### Feature Category 1: Line Movement & Market Efficiency

**Goal:** Capture sharp money vs public money dynamics

| Feature | Type | Description | Leakage Risk | Data Source |
|---------|------|-------------|--------------|-------------|
| `opening_line` | Continuous | Opening spread (Sunday/Monday lookahead) | ⚠️ MEDIUM | Sportsbooks, Odds APIs |
| `current_line` | Continuous | Line as of Friday 6pm ET (48hr before kickoff) | ⚠️ MEDIUM | Sportsbooks, Odds APIs |
| `line_movement` | Continuous | current_line - opening_line | ⚠️ MEDIUM | Calculated |
| `line_movement_direction` | Binary | 1 = moved toward favorite, -1 = toward dog | ⚠️ MEDIUM | Calculated |
| `reverse_line_movement` | Binary | Line moved opposite of public betting % | ⚠️ HIGH | Requires public % data |
| `line_freeze` | Binary | No movement in last 48 hours despite volume | ⚠️ HIGH | Requires tick data |

**Leakage Constraints:**
- ✅ **Safe:** Opening line (published days before game)
- ⚠️ **Risky:** Current line must be frozen at specific cutoff (e.g., Friday 6pm ET)
- ❌ **Unsafe:** Closing line (often moves in final hours based on late info)

**Implementation Notes:**
- Need **timestamped line data** to enforce cutoff
- Opening line typically available Sunday/Monday for next week's games
- Must decide: "Friday 6pm cutoff" or "72 hours pre-kickoff" or similar
- Risk: Some games don't have opening lines until later in week

**Hypothesis:**
- Lines that move significantly (>1 point) often indicate sharp money
- Reverse line movement (RLM) is strong contrarian indicator
- Line freezes might signal sportsbook confidence in number

### Feature Category 2: Public Betting Sentiment

**Goal:** Identify overvalued/undervalued teams due to public bias

| Feature | Type | Description | Leakage Risk | Data Source |
|---------|------|-------------|--------------|-------------|
| `public_bet_pct` | Continuous | % of bets on favorite (0-100) | ⚠️ HIGH | Action Network, Covers |
| `public_money_pct` | Continuous | % of money on favorite (0-100) | ⚠️ HIGH | Action Network, pregame.com |
| `sharp_money_indicator` | Binary | Money % >> Bet % (whales on underdog) | ⚠️ HIGH | Calculated |
| `steam_move` | Binary | Rapid line movement (>0.5pt in <1hr) | ❌ VERY HIGH | Requires real-time data |
| `public_fade_opportunity` | Binary | Public >70% on one side, line stable | ⚠️ HIGH | Calculated |

**Leakage Constraints:**
- ⚠️ **Risky:** Public betting % often published Thursday-Saturday
- ⚠️ **Risky:** Need to enforce "Friday 6pm cutoff" or earlier
- ❌ **Unsafe:** Real-time steam moves (late-breaking info)

**Data Availability Assessment:**
- **Action Network:** Publishes public bet % for major games (usually Thursday)
- **Covers.com:** Publishes consensus picks (daily updates)
- **Pregame.com:** Publishes public money % (subscription required)
- **Risk:** Data might not be available for all games, all weeks

**Hypothesis:**
- Heavy public betting (>70%) creates value on underdog
- Sharp money indicator (money % > bet %) is strong signal
- Contrarian betting against public has historical edge

### Feature Category 3: Situational & Contextual Factors

**Goal:** Capture non-performance situational edges

| Feature | Type | Description | Leakage Risk | Data Source |
|---------|------|-------------|--------------|-------------|
| `div_game_rivalry` | Binary | Divisional rivalry (e.g., Cowboys-Eagles) | ✅ NONE | NFL schedule |
| `prime_time_game` | Binary | SNF, MNF, TNF | ✅ NONE | NFL schedule |
| `home_underdog` | Binary | Home team is underdog (historically +EV) | ✅ NONE | Calculated |
| `short_week` | Binary | Thursday game (3-day rest) | ✅ NONE | NFL schedule |
| `long_rest` | Binary | Post-bye week or 10+ day rest | ✅ NONE | NFL schedule |
| `lookahead_game` | Binary | Opponent has strong next-week matchup | ⚠️ LOW | NFL schedule analysis |
| `revenge_game` | Binary | Lost to opponent earlier this season | ⚠️ LOW | Historical matchups |
| `playoff_implications` | Ordinal | 0=none, 1=minor, 2=major, 3=elimination | ⚠️ MEDIUM | Standings-based |

**Leakage Constraints:**
- ✅ **Safe:** All schedule-based features known weeks in advance
- ⚠️ **Risky:** Playoff implications (depends on current standings, which change weekly)

**Hypothesis:**
- Home underdogs are historically undervalued (73-74 ATS since 2003)
- Divisional games are more unpredictable (lower favorites cover rate)
- Lookahead games create letdown spots (team looking past opponent)
- Revenge games slightly overvalued (public narrative > actual edge)

### Feature Category 4: Injury & Roster Dynamics

**Goal:** Capture late-breaking injury impact not in power ratings

| Feature | Type | Description | Leakage Risk | Data Source |
|---------|------|-------------|--------------|-------------|
| `qb_injury_status` | Ordinal | 0=healthy, 1=questionable, 2=out, 3=backup | ⚠️ MEDIUM | NFL injury reports |
| `key_player_out` | Binary | Star player (QB/WR1/EDGE) ruled out | ⚠️ MEDIUM | NFL injury reports |
| `injury_report_delta` | Continuous | # of new injuries since Wednesday | ⚠️ HIGH | NFL injury reports |
| `late_scratch` | Binary | Key player ruled out Fri/Sat | ❌ VERY HIGH | NFL injury reports |

**Leakage Constraints:**
- ⚠️ **Risky:** Injury reports updated daily (Wed/Thu/Fri)
- ⚠️ **Risky:** Must freeze at Friday final report (official NFL deadline)
- ❌ **Unsafe:** Late scratches (Saturday/Sunday morning) contain insider info

**Data Availability:**
- **NFL.com:** Official injury reports (Wed/Thu/Fri)
- **FantasyPros:** Aggregated injury updates
- **Risk:** Subjective interpretation (what is "key player"?)

**Hypothesis:**
- QB injuries have largest spread impact (already in power ratings via QB_Adj)
- Multiple injuries on one side create compounding effect
- Late-week injury downgrades (Q→OUT) often not priced in

### Feature Category 5: Environmental & External Factors

**Goal:** Weather and venue impacts not in team metrics

| Feature | Type | Description | Leakage Risk | Data Source |
|---------|------|-------------|--------------|-------------|
| `temp_gameday` | Continuous | Temperature at kickoff (Fahrenheit) | ⚠️ MEDIUM | Weather APIs |
| `wind_speed` | Continuous | Wind speed in MPH | ⚠️ MEDIUM | Weather APIs |
| `precipitation` | Ordinal | 0=clear, 1=rain, 2=snow | ⚠️ MEDIUM | Weather APIs |
| `dome_game` | Binary | Indoor stadium (no weather) | ✅ NONE | Venue database |
| `altitude` | Binary | Denver (5,280 ft elevation) | ✅ NONE | Venue database |
| `grass_vs_turf` | Binary | Grass=1, Turf=0 | ✅ NONE | Venue database |
| `crowd_noise` | Continuous | Expected attendance % (playoff implications) | ⚠️ LOW | Calculated |

**Leakage Constraints:**
- ⚠️ **Risky:** Weather forecasts get more accurate closer to game
- ⚠️ **Risky:** Must freeze forecast at 48-hour mark
- ✅ **Safe:** Venue characteristics known in advance

**Hypothesis:**
- Wind >15mph reduces passing offense (favors rush-heavy teams)
- Snow favors physical run-first teams (historically)
- Denver altitude affects visiting teams (thin air, fatigue)
- Dome teams underperform in outdoor cold games (adaptation)

---

## Data Requirements & Availability

### Critical Dependencies (Must Have)

| Data Type | Source Options | Availability | Cost | Leakage Risk |
|-----------|---------------|--------------|------|--------------|
| Opening Lines | OddsAPI, ActionNetwork | ✅ Good | Free-$50/mo | ⚠️ MEDIUM |
| Current Lines | OddsAPI, ActionNetwork | ✅ Good | Free-$50/mo | ⚠️ MEDIUM |
| NFL Schedule | nflverse, ESPN API | ✅ Excellent | Free | ✅ NONE |
| Injury Reports | NFL.com, FantasyPros | ✅ Good | Free-$10/mo | ⚠️ MEDIUM |
| Weather Forecasts | OpenWeatherMap, Weather.gov | ✅ Good | Free | ⚠️ MEDIUM |

**Assessment:** ✅ **REALISTIC** - All critical data available with free or low-cost sources.

### Nice-to-Have (Optional)

| Data Type | Source Options | Availability | Cost | Leakage Risk |
|-----------|---------------|--------------|------|--------------|
| Public Betting % | Action Network, Covers | ⚠️ Partial | $20-$100/mo | ⚠️ HIGH |
| Sharp Money Indicators | Pregame.com, BetLabs | ⚠️ Limited | $100-$300/mo | ❌ VERY HIGH |
| Steam Moves | Real-time odds feeds | ❌ Rare | $500+/mo | ❌ VERY HIGH |
| Social Sentiment | Twitter API, Reddit scraping | ⚠️ Partial | Free-$100/mo | ⚠️ HIGH |

**Assessment:** ⚠️ **CHALLENGING** - Public betting data often expensive, incomplete, or delayed.

**Recommendation for v1.3:**
- **Start with free/low-cost sources only** (opening lines, schedule, weather, injury reports)
- **Avoid expensive data** (sharp money, steam moves) until v1.2 proves profitable
- **Phase 2 expansion:** Add public betting % if v1.3 shows promise

---

## Training & Validation Strategy

### Time-Series Cross-Validation (Required)

**Why:** NFL data has temporal dependencies (teams improve/decline, market adapts)

**Approach: Rolling Window CV**

```
Season 2020:
  Train: Weeks 1-8   → Test: Weeks 9-10
  Train: Weeks 1-10  → Test: Weeks 11-12
  Train: Weeks 1-12  → Test: Weeks 13-14
  ...

Season 2021:
  Train: Weeks 1-8   → Test: Weeks 9-10
  ...

Season 2024:
  Train: Weeks 1-10  → Test: Weeks 11-18
```

**Metrics by Fold:**
- MAE vs Vegas (v1.3 vs v1.2 vs v1.0)
- ATS accuracy at different edge thresholds
- Feature importance stability across folds

### Leakage Detection Protocol

**Critical Validations:**

1. **Timestamp Audit:**
   - Log exact timestamp when each feature was "available"
   - Verify all features frozen ≥48 hours before kickoff
   - Flag any features that updated closer to game time

2. **Feature-Target Correlation Check:**
   - If any market feature has r > 0.8 with Vegas line → likely leakage
   - Example: `current_line` should NOT correlate >0.9 with `vegas_line`
   - Residual features (line_movement) should have weaker correlation

3. **Holdout Season Test:**
   - Train on 2020-2023 only
   - Test on entire 2024 season (all 18 weeks)
   - If 2024 performance >> 2023 → possible leakage (market adapted)

4. **Feature Ablation:**
   - Remove market features one at a time
   - If removing `feature_X` drops accuracy >5% → likely leakage or overfitting

### Hyperparameter Tuning

**Grid Search Candidates:**

| Parameter | Values to Test |
|-----------|----------------|
| Regularization (alpha) | [1.0, 5.0, 10.0, 20.0, 50.0, 100.0] |
| Model Type | [Ridge, Lasso, ElasticNet, GradientBoosting] |
| Training Window | [5 weeks, 8 weeks, 10 weeks, full season] |
| Feature Set | [All, Schedule-only, Market-only, Hybrid] |

**Evaluation:**
- 5-fold time-series CV
- Select hyperparameters that maximize **out-of-sample ATS accuracy**
- Penalize models with high variance across folds (stability matters)

---

## Interface with v1.2 Outputs

### Input Schema for v1.3

v1.3 will receive **two types of inputs**:

#### 1. v1.2 Outputs (Predictions)

```python
v1_2_outputs = {
    'base_prediction': float,      # v1.0 deterministic output
    'correction_1': float,          # v1.2 learned correction
    'v1_2_prediction': float,       # base + correction_1
    'vegas_line': float,            # Training target only
}
```

#### 2. Market Intelligence Features

```python
market_features = {
    # Line movement
    'opening_line': float,
    'current_line': float,
    'line_movement': float,

    # Situational
    'div_game_rivalry': bool,
    'prime_time_game': bool,
    'home_underdog': bool,
    'short_week': bool,

    # Weather
    'temp_gameday': float,
    'wind_speed': float,
    'precipitation': int,

    # Injury (optional)
    'qb_injury_status': int,
    'key_player_out': bool,
}
```

### Training Pipeline Integration

```python
from ball_knower.models.v1_2_correction import SpreadCorrectionModel
from ball_knower.models.v1_3_market_meta import MarketMetaModel  # NEW

# Step 1: Train v1.2 (as before)
v1_2_model = SpreadCorrectionModel(base_model=v1_0_model)
v1_2_model.fit(train_matchups, vegas_lines)

# Step 2: Get v1.2 predictions for training set
v1_2_train_predictions = v1_2_model.predict(train_matchups)

# Step 3: Calculate residual_2 (what v1.2 missed)
residual_2 = vegas_lines - v1_2_train_predictions

# Step 4: Train v1.3 on market features + residual_2
market_features_train = extract_market_features(train_matchups, cutoff="Friday 6pm")
v1_3_model = MarketMetaModel()
v1_3_model.fit(market_features_train, residual_2)

# Step 5: Predict with full stack
v1_2_test_predictions = v1_2_model.predict(test_matchups)
market_features_test = extract_market_features(test_matchups, cutoff="Friday 6pm")
v1_3_corrections = v1_3_model.predict(market_features_test)
v1_3_final_predictions = v1_2_test_predictions + v1_3_corrections
```

### Modular Architecture

```python
class BallKnowerStack:
    """Full v1.0 → v1.2 → v1.3 prediction stack."""

    def __init__(self, enable_v1_2=True, enable_v1_3=False):
        self.v1_0_model = DeterministicSpreadModel()
        self.v1_2_model = SpreadCorrectionModel(base_model=self.v1_0_model) if enable_v1_2 else None
        self.v1_3_model = MarketMetaModel() if enable_v1_3 else None

    def predict(self, matchups, market_data=None):
        # Layer 1: v1.0 base
        predictions = self.v1_0_model.predict_batch(matchups)

        # Layer 2: v1.2 correction (if enabled)
        if self.v1_2_model:
            predictions = self.v1_2_model.predict(matchups)

        # Layer 3: v1.3 market meta (if enabled + data available)
        if self.v1_3_model and market_data is not None:
            corrections_3 = self.v1_3_model.predict(market_data)
            predictions = predictions + corrections_3

        return predictions
```

**Benefits:**
- ✅ Can toggle v1.2 and v1.3 layers independently
- ✅ Graceful degradation if market data unavailable
- ✅ Easy A/B testing: v1.2-only vs v1.2+v1.3

---

## Risk Assessment

### High-Risk Issues (Must Address Before Implementation)

#### 1. Temporal Leakage (Line Movement)

**Risk:** Current line contains information not available 48hrs before game

**Example:**
- Friday 6pm: Line is DAL -3
- Saturday 2pm: Star QB ruled OUT, line moves to DAL +2
- Sunday 12pm: We predict using DAL +2 (leakage!)

**Mitigation:**
- **Freeze cutoff:** All market features frozen at Friday 6pm ET sharp
- **Timestamp logging:** Record when each feature was captured
- **Validation:** If feature correlates >0.9 with Vegas line → likely leakage
- **Manual audit:** Spot-check 10 games per season for late-line moves

#### 2. Public Betting Data Availability

**Risk:** Public betting % data inconsistent, delayed, or paywalled

**Example:**
- Action Network: Only publishes for primetime games (SNF/MNF)
- Covers: Publishes Thursday, but data might be from Wednesday
- Pregame.com: Requires $300/year subscription

**Mitigation:**
- **Phase 1:** Build v1.3 WITHOUT public betting features
- **Phase 2:** Add public betting only if free data becomes available
- **Fallback:** Use only opening line, line movement, and schedule features

#### 3. Feature Availability Across Seasons

**Risk:** Market data sources change, go offline, or change pricing

**Example:**
- 2024: OddsAPI offers free tier (100 calls/day)
- 2025: OddsAPI removes free tier, requires $100/month
- Model breaks if data source disappears

**Mitigation:**
- **Multi-source:** Use 2-3 data sources for critical features (line movement)
- **Historical backup:** Download and archive all data locally
- **Graceful degradation:** Model falls back to v1.2 if v1.3 data unavailable

### Medium-Risk Issues (Monitor and Mitigate)

#### 4. Overfitting to Market Noise

**Risk:** Model learns spurious correlations (e.g., "Thursdays favor home underdogs")

**Mitigation:**
- **Strong regularization:** Start with alpha=50.0 (higher than v1.2's 10.0)
- **Feature selection:** Use Lasso to zero out weak features
- **Cross-validation:** Require stable performance across multiple seasons
- **Domain validation:** Manually review top features for reasonableness

#### 5. Market Adaptation

**Risk:** Betting market adapts to exploit patterns v1.3 finds

**Example:**
- 2020-2023: Home underdogs cover at 54% (edge found)
- 2024: Sportsbooks adjust lines, home underdogs now 50% (edge gone)

**Mitigation:**
- **Rolling retraining:** Retrain v1.3 every 4-6 weeks during season
- **Performance monitoring:** Track ATS accuracy by week; alert if drops <52%
- **Feature decay:** Downweight features that lose predictive power over time

#### 6. Sample Size Limitations

**Risk:** NFL has only ~270 games per season → limited training data

**Example:**
- `prime_time_home_underdog_in_snow` → might occur 2x per season
- Model can't learn reliable patterns from N=2

**Mitigation:**
- **Feature aggregation:** Avoid hyper-specific combinations
- **Pooling:** Train on 2019-2024 combined (6 seasons, ~1600 games)
- **Bayesian priors:** Regularize rare events toward league average

### Low-Risk Issues (Acceptable Trade-offs)

#### 7. Weather Forecast Uncertainty

**Risk:** 48-hour weather forecasts sometimes wrong

**Mitigation:** ✅ **ACCEPT** - Forecast errors are random, not systematic bias

#### 8. Subjectivity in Feature Engineering

**Risk:** "Lookahead game" definition is subjective

**Mitigation:** ✅ **ACCEPT** - Use clear rules (e.g., next opponent is top-5 team)

---

## Success Criteria & Exit Conditions

### Go/No-Go Decision After v1.2 Validation

**Proceed with v1.3 development IF:**
- ✅ v1.2 achieves MAE < 2.5 points vs Vegas on 2024 test weeks
- ✅ v1.2 achieves ATS accuracy >52% at edge ≥1.0 point
- ✅ v1.2 feature importance makes sense (no extreme coefficients)
- ✅ Free market data sources are reliable and available

**CANCEL v1.3 IF:**
- ❌ v1.2 fails to beat v1.0 baseline (no improvement from ML)
- ❌ v1.2 shows severe overfitting (train MAE << test MAE)
- ❌ Market data sources require >$100/month for necessary features

### v1.3 Prototype Success Criteria (6-Week Development)

**Minimum Viable v1.3:**
- [ ] Achieves MAE ≤ v1.2 MAE (no regression)
- [ ] ATS accuracy ≥ v1.2 at edge ≥1.0 point
- [ ] Passes leakage validation (timestamp audit, correlation checks)
- [ ] Runs on historical 2024 data without errors
- [ ] Feature importance is interpretable (no "magic" features)

**Strong v1.3 Performance:**
- [ ] MAE improvement >0.1 points vs v1.2
- [ ] ATS accuracy >53% at edge ≥1.0 point
- [ ] ROI estimate >2% at edge ≥2.0 points
- [ ] Stable performance across 2020-2024 seasons

**Exceptional v1.3 Performance (Unlikely):**
- [ ] MAE improvement >0.3 points vs v1.2
- [ ] ATS accuracy >55% at edge ≥1.0 point
- [ ] ROI estimate >5% at edge ≥2.0 points

### Exit Conditions (Stop Development)

**Abandon v1.3 IF:**
- ❌ After 50 hours of development, no improvement over v1.2
- ❌ Leakage detected that cannot be fixed
- ❌ Data sources become unavailable or too expensive
- ❌ Market features show zero predictive power (all coefficients near 0)

---

## Implementation Roadmap (Proposed)

### Phase 0: Data Collection (Weeks 1-2)

- [ ] Identify free/low-cost data sources for each feature category
- [ ] Build data ingestion pipeline for opening lines (OddsAPI)
- [ ] Build data ingestion pipeline for line movement tracking
- [ ] Archive 2020-2024 historical lines (if available)
- [ ] Test weather API integration (OpenWeatherMap)
- [ ] Document data schemas and cutoff times

### Phase 1: Minimal Viable v1.3 (Weeks 3-4)

**Features: Schedule + Line Movement Only (No Public Betting)**

- [ ] Implement `extract_market_features()` function
  - opening_line, current_line, line_movement
  - div_game_rivalry, prime_time_game, home_underdog, short_week
  - dome_game, altitude (venue characteristics)

- [ ] Create `ball_knower/models/v1_3_market_meta.py`
  - `MarketMetaModel` class (Ridge regression)
  - Fits on residual_2 (Vegas - v1_2_prediction)
  - Uses ONLY market features (no team performance)

- [ ] Test on 2024 data
  - Train on weeks 1-10
  - Test on weeks 11-18
  - Compare MAE, ATS accuracy vs v1.2

### Phase 2: Add Weather & Injury Features (Weeks 5-6)

- [ ] Integrate weather API (temp, wind, precipitation)
- [ ] Scrape NFL injury reports (Friday final reports)
- [ ] Add weather features to feature extraction
- [ ] Add injury features (qb_injury_status, key_player_out)
- [ ] Re-test on 2024 data

### Phase 3: Validation & Hyperparameter Tuning (Weeks 7-8)

- [ ] Implement time-series cross-validation (2020-2024)
- [ ] Grid search: alpha, model type, training window
- [ ] Leakage detection: timestamp audit, correlation checks
- [ ] Feature importance analysis
- [ ] Stability testing (remove features, check robustness)

### Phase 4: Backtest & Documentation (Weeks 9-10)

- [ ] Full backtest on 2020-2024 (5 seasons)
- [ ] Create CLI tool: `scripts/run_v1_3_market_backtest.py`
- [ ] Write documentation: `docs/V1_3_MARKET_INTELLIGENCE.md`
- [ ] Create performance comparison report (v1.0 vs v1.2 vs v1.3)
- [ ] Merge to main if success criteria met

### Phase 5: Live Deployment (Post-Validation)

- [ ] Real-time data pipeline for weekly predictions
- [ ] Monitoring dashboard (track ATS accuracy by week)
- [ ] Alerting system (if accuracy drops <52%, investigate)
- [ ] Weekly retraining (update model with latest games)

---

## Open Questions (To Resolve Before Implementation)

### Data Strategy

1. **Line movement cutoff time:** Friday 6pm ET or 48 hours pre-kickoff?
   - Friday 6pm is consistent, but some games are Thu/Sat/Sun/Mon
   - 48 hours works for all games, but variable timestamp

2. **Public betting data:** Pursue paid sources or skip entirely?
   - Action Network: $30/month (limited coverage)
   - Pregame.com: $300/year (full coverage, but expensive)
   - Skip for Phase 1, revisit if v1.3 shows promise?

3. **Historical data:** How far back to train?
   - 2020-2024 (5 seasons, COVID era forward)
   - 2015-2024 (10 seasons, more data but older market dynamics)

### Model Architecture

4. **Residual stacking vs ensemble:** Stick with Option A or try Option B?
   - Option A (recommended): v1.3 learns residual_2 only
   - Option B: v1.3 ensemble combines v1.0, v1.2, market features

5. **Model type:** Ridge, Lasso, ElasticNet, or GradientBoosting?
   - Ridge (recommended): Consistent with v1.2, good for correlated features
   - Lasso: Automatic feature selection, but might be too aggressive
   - GradientBoosting: More flexible, but higher overfitting risk

6. **Feature interactions:** Allow polynomial/interaction terms?
   - Example: `line_movement × home_underdog`
   - Risk: Overfitting with limited NFL data

### Validation

7. **Leakage tolerance:** What correlation threshold triggers concern?
   - Proposed: If market feature correlates >0.8 with Vegas line → leakage
   - Too strict? Too lenient?

8. **Success threshold:** What improvement justifies v1.3?
   - Proposed: MAE improvement ≥0.1 points AND ATS accuracy ≥52.5%
   - Or should bar be higher (MAE ≥0.2, ATS ≥53%)?

---

## Comparison: v1.2 vs v1.3 (Proposed)

| Dimension | v1.2 Correction | v1.3 Market Meta |
|-----------|-----------------|------------------|
| **Input Features** | Team performance (canonical) | Market + situational only |
| **Training Target** | Residual_1 (Vegas - v1.0) | Residual_2 (Vegas - v1.2) |
| **Data Sources** | nfelo, substack, nflverse | OddsAPI, weather, injury reports |
| **Leakage Risk** | ✅ LOW (all pre-game team stats) | ⚠️ MEDIUM (market timing critical) |
| **Feature Count** | ~9 features | ~15-20 features (estimated) |
| **Interpretability** | Medium (Ridge coefficients) | Low (market signals harder to explain) |
| **Stability** | High (team metrics stable) | Medium (market adapts) |
| **Data Cost** | ✅ FREE | ⚠️ $0-$50/month |
| **Overfitting Risk** | Low (regularized, limited features) | Medium (more features, market noise) |

---

## Next Steps (Post-Design Review)

1. **User Review:** Get feedback on design before starting implementation
2. **Data Source POC:** Test OddsAPI, verify free tier sufficient
3. **v1.2 Validation:** Complete v1.2 backtest before committing to v1.3
4. **Go/No-Go Decision:** Proceed with v1.3 only if v1.2 succeeds
5. **Phased Development:** Start with Phase 1 (minimal viable v1.3)

---

## Conclusion

**v1.3 is a HIGH-RISK, HIGH-REWARD addition to Ball Knower.**

**Potential Upside:**
- Capture market inefficiencies (public bias, line movement signals)
- Improve ATS accuracy beyond pure team-performance models
- Learn which situational spots provide edge

**Potential Downside:**
- Temporal leakage if market data not carefully timestamped
- Overfitting to market noise (spurious correlations)
- Data sources become unavailable or expensive
- Market adaptation erodes any edge found

**Recommendation:**
- ✅ **Proceed with design** (this document)
- ⚠️ **Wait for v1.2 validation** before starting implementation
- ✅ **Start minimal** (Phase 1 only: schedule + line movement)
- ⚠️ **Strong leakage controls** (timestamp audit mandatory)
- ⚠️ **Exit early if no signal** (don't force v1.3 if market features weak)

**Design Confidence:** 75%

The architecture is sound and the features are well-researched. The main uncertainty is **data availability** and **leakage risk**. If we can secure reliable, timestamped market data, v1.3 has a good chance of improving over v1.2.

---

**Design Version:** 1.0 DRAFT
**Last Updated:** 2025-11-17
**Status:** Awaiting user review and v1.2 validation
**Next Review:** After v1.2 backtest completes
