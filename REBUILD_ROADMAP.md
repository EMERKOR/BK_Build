# Ball Knower Rebuild Roadmap
**Based on First Principles Review - 2025-11-17**

---

## Overview

This roadmap implements the **Edge Detection** strategy identified in the first principles review.

**Core Strategy:** Stop trying to beat Vegas generally. Find specific scenarios where we have information/methodology edges.

---

## Phase 1: Edge Discovery (1-2 weeks)

### Goal
Identify ≥1 systematic Vegas error in historical data

### Experiments to Run

#### Experiment 1A: Injury Impact Analysis
**Hypothesis:** Vegas over/underreacts to specific types of injuries

**Data needed:**
- injuries.parquet (injury reports)
- player_stats_week.parquet (player value)
- schedules.parquet (lines + outcomes)
- rosters_weekly.parquet (backup quality)

**Analysis steps:**
1. Filter to games with key injuries (QB, top WR, top RB, OL starters)
2. For each injury type, calculate:
   - Vegas line movement after injury news
   - Actual game outcome
   - Vegas error (prediction - actual)
3. Group by injury type and player importance
4. Test hypothesis: `vegas_error ~ injury_type + player_importance + backup_quality`

**Success criteria:**
- Find injury scenario with systematic Vegas error
- >52.4% win rate betting opposite to Vegas adjustment
- n ≥ 30 historical games

**Implementation:**
```python
# File: experiments/exp_1a_injury_impact.py
# 1. Load injury data
# 2. Merge with game outcomes
# 3. Calculate Vegas adjustments
# 4. Identify patterns
# 5. Backtest betting strategy
```

**Deliverable:** `output/exp_1a_injury_report.md` with findings

---

#### Experiment 1B: Weather Edge Analysis
**Hypothesis:** High wind games → totals too high (passing impact undervalued)

**Data needed:**
- schedules.parquet (weather data)
- team_stats_week.parquet (pass/rush splits)

**Analysis steps:**
1. Extract weather data (wind speed, temperature)
2. Filter to extreme weather games:
   - High wind: >15 mph
   - Cold: <25°F
   - Precipitation: heavy snow/rain
3. Compare Vegas total to actual total
4. Test patterns:
   - Wind → under
   - Cold + dome team → under
   - Cold + outdoor team → ??

**Success criteria:**
- Weather scenario with systematic under/over pattern
- >52.4% win rate
- n ≥ 30 games

**Implementation:**
```python
# File: experiments/exp_1b_weather_edge.py
```

**Deliverable:** `output/exp_1b_weather_report.md`

---

#### Experiment 1C: Referee Tendency Analysis
**Hypothesis:** Some referee crews systematically affect totals/spreads

**Data needed:**
- officials.parquet (crew assignments)
- schedules.parquet (games + outcomes)

**Analysis steps:**
1. Calculate per-crew statistics:
   - Average total points (crew games vs league average)
   - Home/away penalty differential
   - Pace (plays per game)
2. Identify outlier crews (>1 std dev from mean)
3. Test if Vegas total adjusts for crew assignment
4. Backtest betting opposite to Vegas if no adjustment

**Success criteria:**
- Referee crew with systematic bias
- Vegas total doesn't account for it
- >52.4% win rate betting based on crew
- n ≥ 30 games

**Implementation:**
```python
# File: experiments/exp_1c_referee_edge.py
```

**Deliverable:** `output/exp_1c_referee_report.md`

---

#### Experiment 1D: Backup QB Performance
**Hypothesis:** Vegas overvalues backup QBs (narrative bias)

**Data needed:**
- rosters_weekly.parquet (QB designations)
- player_stats_week.parquet (QB performance)
- schedules.parquet (lines + outcomes)
- espn_qbr_week.parquet (QB ratings)

**Analysis steps:**
1. Identify games with backup QB starting
2. Calculate Vegas adjustment vs starter (line movement)
3. Compare to actual performance difference
4. Test if Vegas overreacts (backup QB spread too high)

**Success criteria:**
- Backup QB scenario with systematic Vegas error
- Opportunity to fade or back backup QB teams
- >52.4% win rate
- n ≥ 30 games

**Implementation:**
```python
# File: experiments/exp_1d_backup_qb.py
```

**Deliverable:** `output/exp_1d_backup_qb_report.md`

---

#### Experiment 1E: Elite Pass Rush vs Weak Pass Protection
**Hypothesis:** Matchup advantages undervalued in spreads

**Data needed:**
- pfr_adv_pass_week.parquet (pass rush win rate, pass block win rate)
- pfr_adv_def_week.parquet (pressure rates)
- schedules.parquet (lines + outcomes)
- team_stats_week.parquet (sack rates)

**Analysis steps:**
1. Quantify pass rush quality (top 25% = elite)
2. Quantify pass protection quality (bottom 25% = weak)
3. Find games with elite pass rush vs weak pass pro
4. Test if spread undervalues this matchup advantage
5. Calculate actual point differential in these matchups

**Success criteria:**
- Extreme matchup scenario with systematic edge
- >52.4% win rate betting on favorable matchup
- n ≥ 20 games (smaller sample ok for extreme matchup)

**Implementation:**
```python
# File: experiments/exp_1e_matchup_edge.py
```

**Deliverable:** `output/exp_1e_matchup_report.md`

---

### Phase 1 Decision Point

**After completing experiments 1A-1E:**

**If ≥1 experiment shows systematic edge:**
- ✅ Proceed to Phase 2 (build model for that edge)

**If 0 experiments show edge:**
- ❌ Conclude: "No edge found in available data"
- Options:
  - Pivot to different data sources (line movement, public betting %)
  - Abandon betting project, keep as research
  - Explore meta-strategies (line shopping, arbitrage)

---

## Phase 2: Model Building (2-3 weeks)

**Only build models for edges discovered in Phase 1!**

### For Each Confirmed Edge:

#### Step 1: Feature Engineering
Build leak-free features specific to the edge scenario

**Example (Injury Edge):**
```python
# src/features_injury.py

def engineer_injury_features(schedules, injuries, player_stats, rosters):
    """
    Engineer injury-specific features:
    - Player importance (WAR-like metric)
    - Backup quality (experience, past performance)
    - Position criticality (QB > WR > RB > OL)
    - Injury severity (Out > Doubtful > Questionable)
    - Time of injury report (days before game)
    """
    # Implementation
    pass
```

#### Step 2: Model Training
Train model on 2015-2023 data, validate on 2024

**Training objective:** Predict Vegas error, not outcomes
```python
# Train on games where edge scenario exists (injuries, weather, etc.)
y_train = vegas_spread - actual_margin  # Vegas error
X_train = edge_specific_features

# Use simple model (Ridge, Logistic Regression)
# Goal: Identify when to bet, not precise predictions
```

#### Step 3: Backtesting
Rigorous backtest with betting simulation

**Requirements:**
```python
# Backtest criteria
min_win_rate = 0.524  # Break-even at -110
min_sample_size = 30  # Statistical significance
min_roi = 0.05  # 5% ROI required
max_drawdown = 0.25  # 25% max drawdown allowed
```

#### Step 4: Validation
Forward test on 2024 data (out-of-sample)

**If model passes:**
- Document model thoroughly
- Save model artifacts
- Create prediction pipeline
- Proceed to ensemble phase

**If model fails:**
- Analyze failure mode
- Refine or discard
- Return to Phase 1 for new edge discovery

---

## Phase 3: Ensemble & Production Prep (1 week)

### Goal
Combine validated models into conservative betting system

### Ensemble Strategy

**Conservative Multi-Signal Approach:**
```python
# Only bet when multiple models agree

def should_bet(game, models):
    """
    Betting logic requiring confirmation from multiple models
    """
    signals = []

    # Injury model
    if has_key_injury(game):
        injury_edge = models['injury'].predict(game)
        if abs(injury_edge) > 2.0:
            signals.append(('injury', injury_edge))

    # Weather model
    if has_extreme_weather(game):
        weather_edge = models['weather'].predict(game)
        if abs(weather_edge) > 1.5:
            signals.append(('weather', weather_edge))

    # Referee model
    if has_referee_bias(game):
        ref_edge = models['referee'].predict(game)
        if abs(ref_edge) > 1.0:
            signals.append(('referee', ref_edge))

    # Decision logic
    if len(signals) >= 2:
        # Multiple models agree → high confidence
        combined_edge = mean([s[1] for s in signals])
        return 'BET', combined_edge, 'high'

    elif len(signals) == 1 and abs(signals[0][1]) > 4.0:
        # Single very strong signal → medium confidence
        return 'BET', signals[0][1], 'medium'

    else:
        # No bet
        return 'NO_BET', 0, None
```

### Bet Sizing

**Conservative Kelly Criterion:**
```python
# Use fractional Kelly (1/4 or 1/6)
# Accounts for model uncertainty

def calculate_bet_size(edge, confidence, bankroll):
    """
    Kelly sizing with safety margins
    """
    # Convert edge to win probability
    win_prob = edge_to_probability(edge, model_uncertainty=2.0)

    # Full Kelly
    kelly_full = kelly_criterion(win_prob, odds=-110)

    # Fractional Kelly based on confidence
    if confidence == 'high':
        kelly_frac = 0.25  # 1/4 Kelly
    elif confidence == 'medium':
        kelly_frac = 0.15  # ~1/6 Kelly
    else:
        kelly_frac = 0.10  # 1/10 Kelly

    bet_size = bankroll * kelly_full * kelly_frac

    # Hard caps
    bet_size = min(bet_size, bankroll * 0.02)  # Never >2% of bankroll
    bet_size = max(bet_size, 0)  # Never negative

    return bet_size
```

### Paper Trading Period

**Before real money:**
1. Generate predictions for Weeks 12-15
2. Track results as if betting (paper trades)
3. Validate system performance
4. Identify any issues (data pipeline, timing, etc.)

**Success criteria for going live:**
- 4 weeks of paper trading
- Win rate >52% on paper trades
- No technical issues
- Confident in process

---

## Phase 4: Production (Ongoing)

### Weekly Workflow

**Tuesday (lines released):**
1. Download latest data
   - nfelo updates
   - Substack updates
   - Injury reports
   - Weather forecasts
   - Referee assignments

2. Run prediction pipeline
   ```bash
   python predict_current_week.py --week 12 --season 2025
   ```

3. Generate bet recommendations
   ```
   Output:
   Week 12 Recommendations (3 bets)

   Game: KC @ BUF
   Models: injury, matchup
   Edge: +2.8 (bet BILLS)
   Confidence: high
   Recommended bet: $250 (2.5% of $10k bankroll)

   Game: LAR @ SF
   Models: weather
   Edge: UNDER 2.1
   Confidence: medium
   Recommended bet: $150 (1.5% of bankroll)

   ...
   ```

4. Place bets (manually at first)
5. Log bets in tracking sheet

**Sunday (results):**
1. Record actual outcomes
2. Update performance metrics
3. Calculate CLV (closing line value)
4. Update bankroll

**Monthly:**
1. Full performance review
2. Re-validate models on new data
3. Check for edge degradation
4. Adjust if needed

### Monitoring & Alerts

**Stop-loss triggers:**
- Down 20% from peak → pause betting, investigate
- 3 consecutive losing weeks → review models
- Win rate drops below 50% over 30 bets → re-validate
- Any model showing systematic failure → disable that model

**Success triggers:**
- Up 20% → consider increasing bankroll allocation
- Win rate >55% over 50 bets → consider higher Kelly fraction
- Consistent performance → add new edge models

---

## Code Structure (Proposed)

```
BK_Build/
├── data/                           # Raw data (existing)
├── src/
│   ├── config.py                   # ✅ Existing
│   ├── team_mapping.py             # ✅ Existing
│   ├── data_loader.py              # ✅ Existing
│   ├── features.py                 # ✅ Existing (expand)
│   ├── features_injury.py          # 🆕 Injury-specific features
│   ├── features_weather.py         # 🆕 Weather-specific features
│   ├── features_referee.py         # 🆕 Referee-specific features
│   ├── features_matchup.py         # 🆕 Matchup-specific features
│   ├── models.py                   # ✅ Existing (expand)
│   ├── model_injury.py             # 🆕 Injury edge model
│   ├── model_weather.py            # 🆕 Weather edge model
│   ├── model_referee.py            # 🆕 Referee edge model
│   ├── model_matchup.py            # 🆕 Matchup edge model
│   ├── ensemble.py                 # 🆕 Multi-model ensemble
│   ├── betting_utils.py            # ✅ Existing (Kelly, EV, etc.)
│   └── tracking.py                 # 🆕 Bet logging and performance
├── experiments/                    # 🆕 Phase 1 experiments
│   ├── exp_1a_injury_impact.py
│   ├── exp_1b_weather_edge.py
│   ├── exp_1c_referee_edge.py
│   ├── exp_1d_backup_qb.py
│   └── exp_1e_matchup_edge.py
├── output/
│   ├── experiments/                # 🆕 Experiment results
│   ├── models/                     # 🆕 Trained model artifacts
│   └── predictions/                # 🆕 Weekly predictions
├── notebooks/
│   └── analysis/                   # 🆕 Ad-hoc analysis notebooks
├── tests/                          # 🆕 Unit tests
├── predict_current_week.py         # ✅ Existing (expand)
├── backtest_ensemble.py            # 🆕 Full system backtest
└── track_performance.py            # 🆕 Performance monitoring
```

---

## Success Metrics

### Phase 1 (Edge Discovery)
- [ ] ≥1 experiment shows systematic edge
- [ ] Win rate >52.4% in backtest
- [ ] Sample size ≥30 games
- [ ] Documented hypothesis + findings

### Phase 2 (Model Building)
- [ ] Model built for each discovered edge
- [ ] Backtest win rate >52.4%
- [ ] Forward test (2024) win rate >52%
- [ ] ROI >5%
- [ ] Max drawdown <25%

### Phase 3 (Ensemble)
- [ ] Multi-model ensemble framework working
- [ ] Conservative bet sizing implemented
- [ ] Paper trading 4+ weeks successful
- [ ] All technical systems functional

### Phase 4 (Production)
- [ ] 20+ real bets placed
- [ ] Win rate >52.4%
- [ ] Positive ROI over 2+ months
- [ ] Disciplined execution (no tilt bets)
- [ ] Continuous monitoring and adjustment

---

## Timeline

**Optimistic (everything works):**
- Phase 1: 2 weeks
- Phase 2: 3 weeks
- Phase 3: 1 week
- Phase 4: Ongoing
- **Total to production:** 6 weeks

**Realistic (some failures, iteration):**
- Phase 1: 3-4 weeks (multiple experiments, some fail)
- Phase 2: 4-6 weeks (model refinement)
- Phase 3: 2 weeks (integration issues)
- Phase 4: Ongoing
- **Total to production:** 10-12 weeks

**Pessimistic (no edge found):**
- Phase 1: 4-6 weeks (exhaustive search)
- **Conclusion:** No systematic edge in available data
- **Outcome:** Research project, not betting system

---

## Resources Needed

**Time commitment:**
- Phase 1-3: 10-15 hours/week
- Phase 4: 2-3 hours/week (weekly predictions + tracking)

**Computational:**
- Current setup sufficient (pandas, sklearn)
- No GPU needed
- Cloud optional (could help with larger backtests)

**Capital:**
- Research phases: $0
- Paper trading: $0
- Production: $5,000-$10,000 recommended starting bankroll

**Data:**
- All data already available locally ✅
- May need line movement data (optional)
- May need public betting percentages (optional, RLM detection)

---

## Risk Management

**Hard stops:**
1. Down 20% from peak → pause, investigate
2. 3 straight losing weeks → model review
3. Win rate <48% over 30 bets → stop betting
4. Any evidence of edge degradation → disable that edge

**Emotional discipline:**
- Stick to system, no manual overrides
- No "revenge betting" after losses
- No increasing bet sizes after wins
- Track all bets, even losers
- Accept variance (short-term losses normal)

**Financial:**
- Never bet more than 2% of bankroll per game
- Never chase losses
- Separate betting bankroll from personal funds
- Be prepared to lose entire bankroll (risk capital only)

---

## Decision: Go or No-Go?

**Proceed to Phase 1 if:**
- [x] You have 10-15 hours/week for next 6-12 weeks
- [x] You're intellectually curious about finding edges
- [x] You're comfortable with systematic research process
- [x] You can accept "no edge found" outcome
- [x] You won't bet until validated edge exists

**Do NOT proceed if:**
- [ ] You need quick profits
- [ ] You're unwilling to stop if no edge found
- [ ] You'll bet based on gut/hunches (not system)
- [ ] You can't afford to lose testing capital
- [ ] You're looking for guaranteed system

---

## Next Action

**If GO:**
1. Create `experiments/` directory
2. Start with Experiment 1A (injury impact)
3. Load injuries.parquet and explore
4. Document findings
5. Decide on Phase 2 based on results

**If NO-GO:**
- Document learnings from project
- Archive codebase for future reference
- Consider alternative applications (DFS, props betting, etc.)

