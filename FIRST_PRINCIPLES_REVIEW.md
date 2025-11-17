# Ball Knower: First Principles Review
**Date:** 2025-11-17
**Purpose:** Comprehensive evaluation before rebuild

---

## Executive Summary

### The Core Problem
We've been building a model to **predict what Vegas will say**, then betting when we **disagree with Vegas**. This is fundamentally flawed because:

1. **Training objective misalignment**: Model learns to match Vegas, not beat them
2. **Divergences are errors**: When model differs from Vegas by 5+ points, it's usually wrong, not insightful
3. **Information disadvantage**: Vegas has more/better/faster data than we do

### Critical Discovery from "v2.0" Testing
The larger the "edge", the worse the win rate:
- 2-3 point edge: 35% win rate (need 52.4%)
- 3-4 point edge: 40% win rate
- 5+ point edge: 18.5% win rate (!!)

**This is backwards.** True edges should perform better at higher thresholds.

### The Good News
- Excellent data infrastructure in place
- 22 high-quality data sources (mostly unused!)
- Leak-free feature engineering framework
- Clean, modular codebase
- We beat Vegas at predictions (14.61 MAE vs 15.41 MAE)

---

## Part 1: What We Have (Current State)

### Models Built

**v1.0 - Deterministic Baseline**
- Inputs: EPA differential, nfelo differential, Substack ratings
- Method: Weighted combination (no ML)
- Output: Spread prediction from home team perspective
- Status: ✅ Working, tested

**v1.1 - Enhanced Features**
- Adds: Rest advantage, recent form, QB adjustments
- Method: Still deterministic
- Status: ⚠️ Framework exists, not fully implemented

**v1.2 - ML Correction Layer**
- Base: v1.1 predictions
- ML: Ridge regression to learn residuals
- Training: Fit on historical data (2009-2024)
- Status: ✅ Trained and tested
- Performance:
  - Test R²: 0.324
  - Test MAE: 2.96 points (predicting Vegas lines)
  - Beat Vegas by 0.79 points at outcome prediction

### Data Sources Currently Used

1. **nfelo historical data** (from greerreNFL)
   - ELO ratings, QB adjustments, situational mods
   - 2009-2024 data available
   - ~4,300 games with complete data

2. **EPA data** (team_week_epa_2013_2024.csv)
   - Offensive/defensive EPA per play
   - Success rates
   - 2013-2024 seasons

3. **Current week projections** (Week 11)
   - nfelo power ratings
   - Substack ratings
   - QB rankings

### Data Sources Available But UNUSED

**Player-Level Data (14MB+ of data we're ignoring!):**
- `player_stats_week.parquet` (14MB) - Every player, every week
- `snap_counts.parquet` (2.4MB) - Who's actually playing
- `injuries.parquet` (1.5MB) - Injury reports with game status
- `rosters_weekly.parquet` (8.1MB) - Roster composition changes

**Advanced Metrics:**
- `espn_qbr_week.parquet` - QB performance ratings
- `pfr_adv_pass_week.parquet` - Advanced passing stats
- `pfr_adv_rush_week.parquet` - Advanced rushing stats
- `pfr_adv_def_week.parquet` - Advanced defensive stats
- `ngs_passing/receiving/rushing.parquet` - Next Gen Stats (tracking data)
- `ftn_charting.parquet` (1.9MB) - Charting data

**Context Data:**
- `officials.parquet` - Referee crews and tendencies
- `trades.parquet` - Trade timing and impact
- `draft_picks.parquet` - Draft capital
- `schedules.parquet` - Weather, stadium, surface, rest days

**Total Available Data: ~40MB across 22 files**
**Currently Using: ~1MB across 2 files (2.5% utilization!)**

---

## Part 2: What's Working

### ✅ Infrastructure Excellence

1. **Clean codebase**
   - Modular design (config, features, models, data_loader)
   - Single source of truth (config.py)
   - Comprehensive team name mapping
   - Proper git structure

2. **Leak-free feature engineering**
   - All rolling features use `.shift(1)`
   - No future information in training
   - Date-based validation
   - This is CRITICAL and working well

3. **Data loading framework**
   - Handles multiple sources (nfelo, Substack, nfl_data_py)
   - Normalizes team names across sources
   - Tested and validated

4. **Model framework**
   - Progression from simple → complex
   - Interpretable baseline before ML
   - Proper train/test splits
   - TimeSeriesSplit for cross-validation

### ✅ Predictive Performance

**v1.2 beats Vegas at predictions:**
- Ball Knower MAE: 14.61 points (actual outcomes)
- Vegas MAE: 15.41 points
- **0.79 point advantage** ✅

This proves the model has signal! We're extracting useful information from EPA + ELO.

### ✅ Professional Betting Framework

- Expected Value (EV) calculations
- Kelly Criterion bet sizing
- Probability calibration
- ROI simulation by edge threshold
- Closing Line Value (CLV) style analysis

Code is production-ready for these concepts.

---

## Part 3: What's Failing

### 🔴 Fundamental Design Flaw

**Training Objective:** Predict Vegas line
**Betting Objective:** Find games where Vegas is wrong

**These are contradictory!**

#### Why This Fails

1. **Model learns to copy Vegas**
   - Training to match Vegas lines (MAE: 2.96 points)
   - Gets very good at it (R² = 0.60)
   - But betting requires DISAGREEING with Vegas

2. **Divergences are usually errors**
   - When model says -3, Vegas says +2 (5-point edge)
   - Is this insight or mistake?
   - Data shows: **Usually a mistake** (18% win rate!)

3. **The Calibration Paradox**
   ```
   Small edges (1-2 pts):  Could be legitimate differences (44% win rate)
   Large edges (5+ pts):   Almost always model errors (18% win rate)
   ```

   **We can't use the edges we're confident in!**

### 🔴 Information Gaps

**What Vegas has that we don't:**
1. **Injury reports** - We have the data file, but not using it!
2. **Line shopping** - Multiple books, arbitrage detection
3. **Sharp money flow** - They see where big bets go
4. **Proprietary models** - Decades of refinement
5. **Real-time updates** - Instant line adjustments
6. **Market efficiency** - Thousands of bettors finding value

**What we could have:**
- Injury impact modeling (data available!)
- Player-level performance (data available!)
- Roster composition (data available!)
- Referee tendencies (data available!)
- Weather effects (in schedules data!)
- Advanced metrics (22 files of data!)

**Currently:** Using 2 aggregate metrics (EPA, ELO) vs Vegas using everything

### 🔴 Model Simplicity vs Data Richness

**What we're using:**
- Team-level EPA (offense, defense)
- Team-level ELO ratings
- Basic situational adjustments (rest, division games)

**What we're ignoring:**
- Individual player performance (who's actually good?)
- Snap counts (who's actually playing?)
- Injuries (who's actually healthy?)
- Matchup-specific factors (WR1 vs CB1)
- Advanced metrics (success rate, CPOE, pressures)
- Referee tendencies (different officiating styles)
- Weather (wind impacts passing, cold impacts scoring)
- Roster changes (trades, practice squad elevations)

**Analogy:** We're predicting stock prices using only market cap and sector, while ignoring earnings, management, competition, and market conditions.

### 🔴 Betting Simulation Failures

**v1.2 Backtest Results (2024):**
- 116 bets placed (2+ point edge threshold)
- Win rate: 34.5%
- ROI: -34.2%
- Lost $4,360 on $12,760 risked

**This is not variance. This is systematic failure.**

---

## Part 4: First Principles Re-Evaluation

### What Are We Actually Trying To Do?

**Primary Goal:** Make money betting on NFL games
**Not:** Predict what Vegas will say
**Not:** Minimize MAE vs Vegas lines
**Not:** Achieve high R²

### What Would Success Look Like?

**Option A: Beat the closing line**
- Bet early, Vegas moves toward us
- Closing Line Value (CLV) positive
- Requires: Fast data, fast models, good books

**Option B: Find true statistical edges**
- Vegas line is accurate on average
- But systematically wrong in specific scenarios
- Requires: Information Vegas doesn't have OR better models

**Option C: Exploit market inefficiencies**
- Overreactions to news
- Public bias (favorites, primetime)
- Small market games (less sharp action)
- Requires: Understanding market psychology

### What Edges Could We Realistically Find?

**Information edges (possible):**
1. **Injury impact modeling**
   - Vegas adjusts for injuries
   - But do they get the magnitude right?
   - We have full injury data to backtest this!

2. **Depth chart analysis**
   - Not just "QB out" but "who's the backup?"
   - Roster composition quality
   - We have weekly rosters + player stats!

3. **Referee tendencies**
   - Some crews call more penalties
   - Impacts pace, scoring, home/away differently
   - We have officials data!

4. **Weather edges**
   - Wind impacts passing more than Vegas accounts for?
   - Cold weather + dome teams
   - Schedules file has weather data

5. **Matchup-specific edges**
   - Elite pass rush vs weak pass blocking
   - Shutdown corner vs WR-dependent offense
   - We have advanced stats for this!

**Model edges (harder):**
1. **Better probability calibration**
   - Vegas is well-calibrated on average
   - But might miss on tails (big favorites/dogs)

2. **Non-linear interactions**
   - EPA × Rest × QB change
   - Division rival + primetime + cold weather
   - Complex patterns Vegas might miss

**Unlikely edges:**
1. **Pure ELO/EPA superiority** - Vegas uses this too
2. **Generic "better model"** - We don't have their data
3. **Beating closing lines consistently** - Market is very efficient

---

## Part 5: Data Inventory & Opportunities

### Full Data Catalog

| File | Size | Records | Time Span | Current Use | Potential Use |
|------|------|---------|-----------|-------------|---------------|
| `team_week_epa_2013_2024.csv` | 841K | ~8,800 | 2013-2024 | ✅ Core features | More granular EPA splits |
| `schedules.parquet` | 492K | ~5,500 | 1999-2024 | ⚠️ Load only | Weather, rest, surface |
| `player_stats_week.parquet` | 14MB | ~140K | 1999-2024 | ❌ None | Star player performance |
| `rosters_weekly.parquet` | 8.1MB | ~90K | 1999-2024 | ❌ None | Depth chart quality |
| `snap_counts.parquet` | 2.4MB | ~60K | 2012-2024 | ❌ None | Who's actually playing |
| `injuries.parquet` | 1.5MB | ~18K | 2009-2024 | ❌ None | **KEY OPPORTUNITY** |
| `ftn_charting.parquet` | 1.9MB | Unknown | Recent | ❌ None | Advanced QB metrics |
| `players.parquet` | 3.2MB | ~27K | All-time | ❌ None | Player metadata |
| `team_stats_week.parquet` | 1.2MB | ~9K | 2002-2024 | ❌ None | Alternative to EPA |
| `espn_qbr_week.parquet` | 282K | ~5K | 2006-2024 | ❌ None | QB performance |
| `ngs_passing.parquet` | 675K | Unknown | 2016+ | ❌ None | Air yards, completion % over expected |
| `ngs_receiving.parquet` | 1.1MB | Unknown | 2016+ | ❌ None | Separation, YAC |
| `ngs_rushing.parquet` | 353K | Unknown | 2016+ | ❌ None | Rush yards over expected |
| `pfr_adv_pass_week.parquet` | 118K | Unknown | Recent | ❌ None | Pressure rate, time to throw |
| `pfr_adv_rush_week.parquet` | 224K | Unknown | Recent | ❌ None | Run block win rate |
| `pfr_adv_def_week.parquet` | 922K | Unknown | Recent | ❌ None | Pass rush, coverage metrics |
| `officials.parquet` | 139K | ~1K | Multi-year | ❌ None | Referee tendencies |
| `draft_picks.parquet` | 614K | ~11K | All-time | ❌ None | Draft capital value |
| `trades.parquet` | 105K | ~1K | All-time | ❌ None | Trade timing impact |
| `teams.parquet` | 17K | 32 | Current | ❌ None | Team metadata |
| `ff_playerids.parquet` | 1.6MB | ~27K | All-time | ❌ None | Player ID mapping |
| `espn_qbr_season.parquet` | 76K | ~800 | 2006-2024 | ❌ None | Season-level QB ratings |

**Total: 22 files, ~40MB, spanning 1999-2024**
**Using: 1-2 files seriously**

### Highest-Value Unused Data

**🥇 Tier 1: Immediate Impact**

1. **injuries.parquet (1.5MB)**
   - Game status (Out, Questionable, Doubtful)
   - Injury timing relative to line movement
   - **Hypothesis:** Vegas overreacts or underreacts to certain injuries
   - **Test:** Backtest line movement after injury news
   - **Edge:** Bet opposite to overreactions

2. **snap_counts.parquet (2.4MB)**
   - Who actually played and how much
   - Correlate playing time with performance
   - **Hypothesis:** Rookies/backups have higher variance than Vegas accounts for
   - **Test:** Model team performance with backup QBs, backup OL, etc.
   - **Edge:** Fade teams relying on backups in key positions

3. **schedules weather data**
   - Wind speed, temperature
   - **Hypothesis:** Wind impacts passing more than totals/spreads account for
   - **Test:** High wind games vs total/spread accuracy
   - **Edge:** Under totals in high wind, fade passing teams

**🥈 Tier 2: Advanced Modeling**

4. **player_stats_week.parquet (14MB)**
   - Individual player performance
   - Build team strength from player components
   - **Hypothesis:** Team is more than sum of averages
   - **Edge:** Better capture of hot/cold streaks

5. **rosters_weekly.parquet (8.1MB)**
   - Roster composition over time
   - Depth chart quality modeling
   - **Hypothesis:** Depth matters when starters are injury-prone
   - **Edge:** Fade thin teams, back deep teams

6. **officials.parquet (139K)**
   - Referee crew assignments
   - Penalty rates, home/away bias
   - **Hypothesis:** Some refs favor road teams or impact totals
   - **Test:** Referee crew vs line accuracy
   - **Edge:** Situational based on referee assignment

**🥉 Tier 3: Marginal/Experimental**

7. **NGS metrics** (ngs_passing, ngs_receiving, ngs_rushing)
   - Tracking data: separation, air yards, RYOE
   - **Hypothesis:** Underlying metrics predict future performance
   - **Edge:** Identify regression candidates

8. **PFR advanced stats** (pfr_adv_*)
   - Pass rush win rate, coverage stats
   - **Hypothesis:** Matchup-specific edges (e.g., great pass rush vs bad OL)
   - **Edge:** Situational betting on mismatches

9. **FTN charting** (ftn_charting.parquet)
   - Manual film study data
   - **Hypothesis:** Charting captures decision-making quality
   - **Edge:** QB quality beyond box score

---

## Part 6: Rebuild Strategy

### Core Principle: Information Edges, Not Model Edges

**New Philosophy:**
1. ✅ **Find information Vegas doesn't have** (or doesn't use well)
2. ✅ **Model specific scenarios** where we have edge
3. ❌ **Don't try to build a "better general model"**

### Proposed Architecture: Multi-Model Ensemble

Instead of one model predicting spreads, build **specialized detectors**:

#### Model 1: Injury Impact Model
**Training objective:** Predict game outcome given injury report
**Data sources:**
- injuries.parquet (game status)
- player_stats_week (player importance)
- rosters_weekly (backup quality)
- snap_counts (actual usage)

**Output:** Injury-adjusted spread
**Bet when:** Vegas line hasn't fully adjusted for injury

**Implementation:**
1. For each game, identify key injuries
2. Quantify player value (WAR-style metric)
3. Quantify backup quality
4. Predict injury impact in points
5. Compare to Vegas adjustment
6. Bet if mismatch > threshold

#### Model 2: Weather Edge Model
**Training objective:** Predict total/spread error in weather games
**Data sources:**
- schedules (weather data)
- team_stats (dome vs outdoor splits)
- player_stats (QB performance in weather)

**Output:** Weather adjustment to total/spread
**Bet when:** Weather impact > Vegas adjustment

**Implementation:**
1. Filter to games with significant weather (wind 15+ mph, temp <20°F)
2. Backtest Vegas total/spread accuracy in these games
3. Identify systematic over/under patterns
4. Build correction model
5. Bet opposite to Vegas error pattern

#### Model 3: Referee Tendency Model
**Training objective:** Predict penalty impact on spread/total
**Data sources:**
- officials.parquet
- schedules (ref crew assignments)
- Historical game data with ref crews

**Output:** Referee-adjusted total
**Bet when:** Referee tendency misalignment with line

**Implementation:**
1. Calculate each referee crew's impact on totals/spreads
2. Identify crews with consistent biases (high/low scoring)
3. Compare to market total
4. Bet when mismatch exists

#### Model 4: Matchup-Specific Edge Model
**Training objective:** Predict game outcome from matchup strengths
**Data sources:**
- pfr_adv_* (pass rush win rate, coverage stats)
- team_stats_week (offensive/defensive strengths)
- player_stats_week (key player matchups)

**Output:** Matchup-adjusted spread
**Bet when:** Specific matchup creates edge (e.g., elite pass rush vs backup OL)

**Implementation:**
1. Identify key matchup dimensions (pass rush vs pass pro, coverage vs WR, etc.)
2. Quantify matchup advantages
3. Predict performance given matchup
4. Compare to Vegas line
5. Bet if significant mismatch

#### Model 5: Market Inefficiency Model
**Training objective:** Identify systematic Vegas errors
**Data sources:**
- All historical lines and outcomes
- Public betting percentages (if available)
- Line movement patterns

**Output:** Scenarios where Vegas is systematically wrong
**Bet when:** Game matches inefficiency pattern

**Examples:**
- Primetime favorites overvalued?
- Division underdogs undervalued?
- Road favorites after bye overvalued?
- Cold weather totals too high?

### Ensemble Logic

**Don't bet unless multiple models agree:**

```
IF (injury_edge > 2 points) AND (weather_edge > 1 point):
    → Bet with high confidence
ELIF (matchup_edge > 3 points) AND (market_inefficiency detected):
    → Bet with medium confidence
ELSE:
    → No bet
```

**Conservative approach:**
- Require confirmation from multiple signals
- Reduces false positives (model errors)
- Increases true positives (real edges)

### Training Objective Shift

**OLD (v1.0-v1.2):**
```python
y_train = vegas_spread  # Predict what Vegas will say
```

**NEW (Multi-model):**
```python
# Model 1: Injury Impact
y_train = actual_margin  # Predict actual outcomes
X_train = injury_features  # Only train on games with injuries

# Model 2: Weather Edge
y_train = vegas_spread - actual_margin  # Predict Vegas error
X_train = weather_features  # Only train on weather games

# Model 3: Referee Tendency
y_train = total_points  # Predict scoring
X_train = referee_features + team_features

# Etc.
```

**Key difference:** Each model trained on **specific subset** where we have information edge.

---

## Part 7: Immediate Next Steps

### Phase 1: Data Exploration (Week 1)

**Goal:** Understand what edges exist in the data

**Tasks:**
1. **Injury Analysis**
   - Load injuries.parquet
   - Merge with schedules and outcomes
   - Analyze Vegas line movement after injury news
   - Identify systematic over/underreactions
   - **Deliverable:** "Injury Impact Report" (which injuries Vegas misvalues)

2. **Weather Analysis**
   - Extract weather from schedules
   - Filter to extreme weather games
   - Compare Vegas total/spread to actual outcomes
   - Identify patterns (wind → lower scoring, cold → ???)
   - **Deliverable:** "Weather Edge Report" (specific weather scenarios with edge)

3. **Referee Analysis**
   - Load officials.parquet
   - Calculate per-crew stats (penalties, totals, home/away splits)
   - Identify crews with strong biases
   - Test if Vegas adjusts for this
   - **Deliverable:** "Referee Tendency Report" (crews to target/avoid)

4. **Matchup Analysis**
   - Load pfr_adv_* files
   - Identify extreme matchups (best pass rush vs worst pass pro)
   - Test if these matchups outperform Vegas expectations
   - **Deliverable:** "Matchup Edge Report" (specific scenarios with edge)

**Success criteria:** Find at least ONE scenario with systematic Vegas error (>52.4% win rate, n>30 games)

### Phase 2: Model Building (Week 2-3)

**Goal:** Build specialized models for discovered edges

**Only build models for edges discovered in Phase 1!**

For each confirmed edge:
1. Build feature engineering pipeline
2. Train model on historical data (2015-2023)
3. Validate on hold-out set (2024)
4. Require: >52.4% win rate, positive EV
5. If fails validation → discard

**Deliverable:** 2-4 specialized models with proven backtest performance

### Phase 3: Ensemble Integration (Week 4)

**Goal:** Combine models with conservative betting logic

1. Build ensemble decision logic
2. Require multiple model agreement
3. Backtest full system on 2024
4. Calculate EV, Kelly sizing, drawdown
5. Forward test on upcoming weeks (paper trading)

**Success criteria:**
- Backtest: >52.4% win rate, positive ROI
- Forward test: 3-4 weeks of consistent performance
- Max drawdown: <20%

### Phase 4: Production (Week 5+)

**Only if Phase 3 succeeds:**
1. Automate data pipeline
2. Weekly prediction workflow
3. Small pilot bets (1% Kelly)
4. Track results meticulously
5. Increase sizing slowly if proven

---

## Part 8: Risk Mitigation

### What Could Still Go Wrong?

**Risk 1: No edge exists**
- Mitigation: Phase 1 exploration finds nothing → Stop project
- Fallback: Research-only, no real betting

**Risk 2: Edge exists but small**
- Mitigation: Require >55% win rate before betting
- Fallback: Micro-betting for validation

**Risk 3: Edge exists in backtest, fails live**
- Mitigation: Mandatory 3-4 week forward test
- Fallback: Back to research if forward test fails

**Risk 4: Edge diminishes over time**
- Mitigation: Continuous monitoring, re-validation
- Fallback: Stop betting if performance degrades

**Risk 5: Bankroll ruin**
- Mitigation: Conservative Kelly (1/4 or less)
- Mitigation: Stop-loss (halt if down 20%)
- Fallback: Preserve capital, regroup

### Success Metrics

**Research Phase (Phase 1):**
- ✅ Find ≥1 scenario with systematic Vegas error
- ✅ Backtest shows >52.4% win rate, n≥30 games
- ❌ If not found → Project conclusion: "No edge identified"

**Model Phase (Phase 2-3):**
- ✅ Backtest ROI >5% (after -110 vig)
- ✅ Forward test 3+ weeks positive
- ❌ If fails → Refine or abandon

**Betting Phase (Phase 4):**
- ✅ 20+ bets with >52.4% win rate
- ✅ Positive ROI over 2+ months
- ❌ If fails → Stop betting, return to research

**Long-term:**
- ROI >5% annually
- Max drawdown <25%
- Continuous edge validation

---

## Part 9: Tools & Infrastructure Needed

### Data Pipeline
- [x] Load historical schedules (already working)
- [x] Load EPA data (already working)
- [ ] Load and parse injuries.parquet
- [ ] Load and parse snap_counts.parquet
- [ ] Load and parse player_stats_week.parquet
- [ ] Load and parse rosters_weekly.parquet
- [ ] Load and parse officials.parquet
- [ ] Extract weather from schedules
- [ ] Build master game-level dataset (all features merged)

### Feature Engineering
- [x] Rolling EPA (leak-free) ✅
- [x] Rest days ✅
- [ ] Injury impact features
- [ ] Weather features
- [ ] Referee features
- [ ] Matchup features (pass rush vs pass pro, etc.)
- [ ] Player-level aggregations (backup quality, etc.)

### Model Training
- [x] Ridge regression (existing) ✅
- [ ] Ensemble framework (multi-model voting)
- [ ] Probability calibration (Platt scaling, isotonic regression)
- [ ] Time-series cross-validation (existing, expand)

### Backtesting
- [x] Basic backtest framework ✅
- [ ] Edge-specific backtesting (injury games, weather games, etc.)
- [ ] Ensemble backtesting
- [ ] Monte Carlo simulation (variance bounds)
- [ ] Drawdown analysis

### Production
- [ ] Automated weekly data refresh
- [ ] Model inference pipeline
- [ ] Bet logging and tracking
- [ ] Performance dashboard
- [ ] Alert system (opportunities, results)

---

## Part 10: Decision Point

### The Fundamental Choice

**Path A: Abandon General Model, Build Edge Detectors**
- Stop trying to predict all games
- Focus on specific scenarios with proven edge
- Bet 10-30 games/season instead of 100+
- Higher confidence, lower volume

**Path B: Improve General Model**
- Keep trying to beat Vegas on all games
- Use all 40MB of data
- Build sophisticated ensemble
- Bet 100+ games/season

**Path C: Hybrid**
- General model for baseline
- Edge detectors for high-confidence bets
- Two-tier system

**Recommendation: Path A (Edge Detectors)**

**Rationale:**
1. We've proven general model approach fails (v1.2 backtest)
2. We have untapped data for specific scenarios (injuries, weather, refs)
3. Edge betting is more defensible (clear information advantage)
4. Lower volume = lower risk, easier to manage
5. Can always expand later if successful

### Required Commitment

**Time:**
- Phase 1: 20-30 hours (data exploration)
- Phase 2: 20-30 hours (model building)
- Phase 3: 10-15 hours (ensemble + backtest)
- Phase 4: 2-3 hours/week (production)

**Capital:**
- Recommended starting bankroll: $5,000-$10,000
- Never bet more than 1-2% per game (Kelly fraction)
- Be prepared to lose 20-25% in drawdown

**Emotional:**
- Losing streaks will happen (variance)
- Must stick to system, no tilt betting
- Long-term mindset (100+ bets to validate)

### Go/No-Go Criteria

**Proceed if:**
- [ ] Committed to full research process (Phases 1-3)
- [ ] Have bankroll for pilot betting
- [ ] Willing to stop if no edge found
- [ ] Comfortable with variance/drawdown
- [ ] Interested in learning, not just winning

**Stop if:**
- [ ] Just want quick profit (this isn't that)
- [ ] Can't afford to lose testing bankroll
- [ ] Unwilling to follow strict process
- [ ] Emotionally reactive to losses
- [ ] Looking for guaranteed system (doesn't exist)

---

## Conclusion

### What We've Learned

1. **Infrastructure is solid** - Codebase, data pipeline, leak prevention all excellent
2. **Data is rich** - 40MB of NFL data, mostly untapped
3. **General model approach failed** - Training to match Vegas doesn't create betting edge
4. **Specific edges might exist** - Injuries, weather, refs, matchups worth exploring
5. **Process matters** - Systematic exploration beats ad-hoc model building

### The Path Forward

**Recommended: Pivot to edge detection approach**

1. Explore specific scenarios (Phase 1)
2. Build models only for proven edges (Phase 2)
3. Ensemble with conservative logic (Phase 3)
4. Paper trade before real money (Phase 4)
5. Bet small, scale slowly if successful

### Final Thought

**We're not competing with Vegas on their terms (general modeling). We're finding specific scenarios where we have an information or methodological edge.**

This is the only sustainable path to profitable betting.

---

**Next Action:** Decide on go/no-go, then proceed to Phase 1 (injury analysis) if go.

