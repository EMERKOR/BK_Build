# Ball Knower Specification

## Purpose

Ball Knower is an NFL betting engine designed to predict game outcomes and identify value in betting markets. The system combines historical data, power ratings, and structural features to generate spread predictions that can be compared against Vegas lines.

The core philosophy is to build a **leak-free, interpretable, and modular** prediction system that:
- Outperforms naive baselines
- Identifies genuine betting edges
- Provides transparent reasoning for predictions
- Maintains strict temporal integrity in training/testing

## Model Versions

Ball Knower evolves through three distinct versions, each building on the previous:

### v1.0 - Actual Margin Prediction (Baseline Football Brain)

**Purpose**: Predict actual game margins from pure football features, with no reference to betting markets.

**Target Variable**: `actual_margin = home_score - away_score`

**Core Features**:
- nfelo rating differential (Elo-based power ratings)
- EPA (Expected Points Added) differentials
- Substack power ratings (offensive/defensive/overall)
- Structural factors: home field advantage, rest days, division games

**Model Type**: Linear regression (deterministic baseline)

**Key Insight**: This version establishes the "football truth" - what the actual game result should be based purely on team strength and context. It ignores what Vegas thinks.

**Training Objective**: Minimize prediction error on actual game margins

**Typical Performance**:
- R² ~ 0.15-0.25 on actual margins
- MAE ~ 10-12 points (inherent variance in NFL games)
- Serves as baseline for all other versions

### v1.2 - Vegas Spread Prediction (Market-Aware Model)

**Purpose**: Predict Vegas closing spreads or learn spread corrections that align with market consensus.

**Target Variable**: `vegas_spread_close` (closing line, from home team perspective)

**Core Features**:
- All v1.0 features (nfelo, EPA, power ratings, structural)
- QB metrics and adjustments
- Seasonal context flags
- Historical line movement patterns (where applicable)

**Model Type**: Ridge regression with regularization

**Key Insight**: Vegas lines incorporate information beyond pure team strength (public perception, injury timing, sharp money). This version learns to predict what the market will settle on, not just what "should" happen.

**Training Objective**: Minimize prediction error on Vegas closing lines

**Typical Performance**:
- R² ~ 0.75-0.85 on Vegas spreads (much higher than v1.0 on actuals)
- MAE ~ 1.5-2.5 points vs closing line
- Identifies discrepancies between model and market

### Spread Correction Variant (v1.2 Alternative)

Instead of predicting the full spread, v1.2 can be trained to predict **corrections**:

```
spread_correction = vegas_line - v1_0_prediction
```

This isolates the "market adjustment" component - the difference between pure football metrics and what Vegas sets.

## Critical Constraints

### 1. No Post-Game Information Leakage

**Rule**: Features must be knowable **before kickoff** of the game being predicted.

**Violations to Avoid**:
- Using same-game stats (e.g., actual_margin, final_score, game_epa)
- Including post-game injury reports
- Using stats that reflect current game outcome

**Enforcement**: All rolling features use `.shift(1)` to exclude current game.

**Example**:
```python
# CORRECT: Uses only past games
team_avg_epa = df.groupby('team')['epa_per_play'].rolling(window=4).mean().shift(1)

# WRONG: Includes current game
team_avg_epa = df.groupby('team')['epa_per_play'].rolling(window=4).mean()
```

### 2. No Future Line Information

**Rule**: When predicting Week N games, only use betting lines available **before** Week N.

**Violations to Avoid**:
- Using closing lines from future weeks
- Including line movements that occurred after prediction time
- Leaking "sharp" line information from later in the season

**Enforcement**: Strict time-based train/test splits with cutoff dates.

### 3. Time-Based Train/Test Splits

**Rule**: Always split data chronologically, never randomly.

**Rationale**:
- NFL changes over time (rules, scoring, strategy)
- Random splits leak future information into training
- Realistic simulation requires predicting truly unseen futures

**Standard Split**:
- Train: Seasons 2009-2024
- Test: Season 2025 (out-of-sample)
- Validation: Last 2-3 weeks of 2024 (for hyperparameter tuning)

**Example**:
```python
# CORRECT: Chronological split
train = df[df['season'] < 2025]
test = df[df['season'] == 2025]

# WRONG: Random split (leaks future into training)
train, test = train_test_split(df, test_size=0.2, random_state=42)
```

### 4. Spread Convention

All spreads follow the **home team perspective**:
- **Negative** = home team favored (e.g., -7 means home favored by 7)
- **Positive** = away team favored (e.g., +3 means away favored by 3)

This convention must be consistent across:
- Vegas lines
- Model predictions
- Edge calculations
- ROI analysis

## Scope and Non-Goals

### In Scope for v1.0-v1.2

- Predicting game spreads (total margin)
- Comparing predictions to Vegas closing lines
- Identifying betting edges (model disagrees with market)
- Backtesting historical performance
- Time-series safe feature engineering
- Team-level aggregates (power ratings, EPA, QB metrics)

### Explicitly Out of Scope

- Predicting totals (over/under)
- Player props (individual player performance)
- Live in-game betting (requires real-time data)
- Same-game parlays or exotic bets
- Moneyline predictions (implied by spread, not modeled separately)

### Future Considerations (v2.0+)

- Ensemble models combining v1.0 and v1.2
- Advanced QB injury adjustments
- Weather impact modeling
- Referee tendencies
- Public betting percentage integration
- Bayesian uncertainty quantification

## Data Requirements

### Minimum Viable Dataset

To train Ball Knower models, you need:

1. **Historical Games** (2009-2024):
   - Game ID, season, week, date
   - Home/away teams
   - Final scores
   - Vegas closing spreads

2. **Team Ratings** (per game):
   - nfelo ratings (Elo-based power metric)
   - EPA offensive/defensive ratings
   - Power rankings from trusted sources

3. **Structural Context**:
   - Home/away/neutral site flags
   - Rest days (days since last game)
   - Division game flags
   - Playoff flags

### Recommended Enhancements

- QB-specific metrics (QBR, EPA, completion %)
- Strength of schedule adjustments
- Recent form (last 3-4 games)
- Line movement data (open vs close)

## Validation and Testing

### Correctness Checks

Before deploying any model:

1. **Check for leakage**: Verify no future information in features
2. **Validate splits**: Confirm train/test are properly time-separated
3. **Inspect distributions**: Ensure feature distributions are reasonable
4. **Test edge cases**: Neutral sites, playoff games, season openers

### Performance Benchmarks

**v1.0 (Actual Margin)**:
- Should beat naive baseline (home wins by 2.5)
- R² > 0.10 on test set
- MAE < 13 points on test set

**v1.2 (Vegas Spread)**:
- Should beat v1.0 on spread prediction
- R² > 0.70 vs Vegas closing lines
- MAE < 3 points vs closing lines
- Positive ROI on edges > 2 points (historically)

## Architecture Philosophy

Ball Knower is designed with these principles:

1. **Interpretability First**: Prefer linear models over black boxes
2. **Deterministic Baseline**: v1.0 requires no training, just weights
3. **Incremental Complexity**: Each version adds one new capability
4. **Leak-Free Guarantee**: Every feature is auditable for temporal validity
5. **Modular Design**: Loaders, features, models are separate concerns

---

**Last Updated**: 2025-11-18
**Maintained By**: Ball Knower Development Team
