# Ball Knower Feature Tiers

This document categorizes features by their role, reliability, and risk of data leakage. Features are organized into tiers to help developers understand which features are safe to use, which require careful validation, and which are explicitly forbidden.

## Tier Definitions

### T0 - Structural Features (Always Safe)

**Definition**: Contextual information about the game that is knowable well in advance and carries zero risk of leakage.

**Characteristics**:
- Available before the season starts (or at schedule release)
- Never changes based on game outcomes
- Provides basic temporal and spatial context

**Examples**:

| Feature | Description | Values |
|---------|-------------|--------|
| `season` | NFL season year | 2009-2025 |
| `week` | Week number in season | 1-18 (regular), 19-22 (playoffs) |
| `is_home` | Home team indicator | 0 (away), 1 (home) |
| `is_neutral` | Neutral site flag | 0 (home/away), 1 (neutral) |
| `rest_days` | Days since last game | 3-14 (typical) |
| `is_division_game` | Same-division matchup | 0 (no), 1 (yes) |
| `is_playoff` | Playoff game flag | 0 (regular), 1 (playoff) |
| `day_of_week` | Game day | 0 (Thu), 4 (Mon), 6 (Sun) |

**Usage Notes**:
- T0 features can be used freely without leakage concerns
- Often interact with other features (e.g., home + rest advantage)
- Provide interpretable baseline effects

**Home Field Advantage**:
- Constant applied: **2.5 points** (configurable in `config.py`)
- Neutral site games: **0 points** home advantage
- Some teams have historically stronger/weaker home advantages (future enhancement)

---

### T1 - Core Model Inputs (Primary Predictors)

**Definition**: Pre-game team strength metrics that are updated weekly and form the foundation of spread predictions.

**Characteristics**:
- Based on historical performance (past games only)
- Updated before kickoff of prediction target
- No look-ahead bias if properly lagged

**Examples**:

| Feature | Description | Source | Typical Range |
|---------|-------------|--------|---------------|
| `nfelo_diff` | nfelo rating differential (home - away) | nfelo | -300 to +300 |
| `qb_adj_elo` | QB-adjusted Elo rating | nfelo | 1300-1700 |
| `off_rating` | Offensive power rating | Substack | 70-130 |
| `def_rating` | Defensive power rating | Substack | 70-130 |
| `overall_rating` | Overall team rating | Substack | 70-130 |
| `epa_off_L3` | Offensive EPA per play (last 3 games) | nflverse | -0.3 to +0.3 |
| `epa_def_L3` | Defensive EPA per play (last 3 games) | nflverse | -0.3 to +0.3 |
| `epa_margin_L5` | EPA differential (last 5 games) | Derived | -0.5 to +0.5 |
| `qb_epa` | QB EPA per play | Substack | -0.2 to +0.4 |
| `qb_completion_pct` | QB completion percentage | Substack | 55-75% |

**Rolling Windows**:
- Standard windows: **3, 5, 10 games**
- Always use `.shift(1)` to exclude current game
- Example: `epa_off_L3` at Week 5 uses Weeks 2-4 data only

**Leakage Prevention**:
```python
# CORRECT: Excludes current game
df['epa_off_L3'] = df.groupby('team')['epa_off'].shift(1).rolling(3).mean()

# WRONG: Includes current game (LEAKAGE!)
df['epa_off_L3'] = df.groupby('team')['epa_off'].rolling(3).mean()
```

**Usage in Models**:
- v1.0: Uses `nfelo_diff`, `epa_margin`, `overall_rating`
- v1.2: Adds all T1 features + QB metrics

---

### T2 - Market Context (Use with Caution)

**Definition**: Betting market information that provides context but must be handled carefully to avoid look-ahead bias.

**Characteristics**:
- Derived from Vegas odds, public betting, or projections
- Can leak future information if not properly time-stamped
- Useful for identifying model disagreement with market

**Examples**:

| Feature | Description | Leakage Risk | Usage |
|---------|-------------|--------------|-------|
| `vegas_line_open` | Opening spread | Low | Safe if from week-of-game open |
| `vegas_line_close` | Closing spread | Medium | **Target variable** for v1.2, not a feature |
| `vegas_total` | Over/under total | Low | Can inform scoring expectations |
| `public_bet_pct` | % of bets on favorite | Medium | Only if timestamped before kickoff |
| `sharp_money_indicator` | Line movement direction | High | Requires careful temporal filtering |
| `projected_spread` | Substack projection | Low | Available before kickoff |
| `projected_total` | Substack total projection | Low | Available before kickoff |

**When to Use**:
- **v1.0**: Do NOT use market features (pure football model)
- **v1.2**: Can use as features IF available before prediction time
- **Edge calculation**: Always compare model prediction to closing line

**Critical Rule**:
- **NEVER use `vegas_line_close` as a feature when predicting spreads**
- It's acceptable as the **target variable** (what we're trying to predict)
- Using it as a feature would create perfect predictions (circular logic)

---

### T3 - Experimental Features (Do Not Rely On)

**Definition**: Advanced or novel features that are not yet validated for production use.

**Characteristics**:
- Interesting in theory, unproven in practice
- May require additional data sources not yet integrated
- High risk of overfitting or instability

**Examples**:

| Feature | Description | Status | Risk |
|---------|-------------|--------|------|
| `coach_win_pct` | Head coach career win % | Available but unused | Unknown predictive value |
| `roster_av` | Team Approximate Value | Data exists, not integrated | Multicollinearity with ratings |
| `weather_severity` | Wind/temp/precipitation score | Partial data | Missing for many games |
| `referee_tendency` | Ref's avg penalty differential | Not implemented | Small sample sizes |
| `travel_distance` | Miles traveled for away team | Not implemented | Weak historical signal |
| `primetime_flag` | Thursday/Monday night game | Easy to add | Effect size unclear |
| `playoff_implications` | Win-and-in scenarios | Hard to quantify | Subjective |

**Recommendation**:
- Include in exploratory analysis
- Do NOT include in production models until validated on holdout data
- Document any experimental features clearly

---

### TX - Forbidden Features (Never Use)

**Definition**: Features that contain post-game information, leaked market data, or obvious leakage.

**Characteristics**:
- Make backtests look amazing but fail in production
- Violate temporal integrity
- Destroy model credibility

**Strictly Prohibited**:

| Forbidden Feature | Why It's Forbidden | Leakage Type |
|-------------------|-------------------|--------------|
| `actual_margin` | Game result | Post-game information |
| `final_score` | Game result | Post-game information |
| `home_score` | Game result | Post-game information |
| `away_score` | Game result | Post-game information |
| `game_epa` | Calculated from game plays | Post-game statistic |
| `game_total_yards` | Accumulated during game | Post-game statistic |
| `turnovers` | Occurred during game | Post-game statistic |
| `time_of_possession` | Game-specific metric | Post-game statistic |
| `vegas_line_close` (as feature) | Circular prediction target | Market leakage |
| `line_from_future_week` | Line from Week N+1 | Temporal leakage |
| `epa_off_L3` (without shift) | Includes current game | Rolling stat leakage |

**Red Flags to Watch For**:
- Feature has perfect correlation with target (R² > 0.95)
- Feature is not available until after kickoff
- Feature uses `.rolling()` without `.shift(1)`
- Feature references future dates/weeks

**Validation Check**:
```python
# Before adding any feature, ask:
# "Could I know this value at 1 PM ET on Sunday, Week N, before any Week N games start?"
# If NO → Do not use it
```

---

## Feature Engineering Best Practices

### Temporal Ordering

1. **Sort by date before any calculations**:
   ```python
   df = df.sort_values(['team', 'season', 'week'])
   ```

2. **Always shift rolling stats**:
   ```python
   df['avg_epa'] = df.groupby('team')['epa'].shift(1).rolling(window=5).mean()
   ```

3. **Fill missing values thoughtfully**:
   - First games of season: Use league average or prior season
   - Missing data: Forward-fill or use sensible defaults (never backfill)

### Feature Scaling

- **nfelo diffs**: Already scaled appropriately (~100 points = ~4-5 point spread)
- **EPA**: Already normalized (mean ~0, std ~0.2)
- **Ratings**: Substack ratings are 0-100 scale, may need normalization
- **Binary flags**: Keep as 0/1 (do not scale)

### Feature Interactions

Useful interaction terms (v2.0+):
- `nfelo_diff × is_playoff` (playoff boost for strong teams)
- `rest_days × travel_distance` (compounding fatigue)
- `qb_epa × opponent_def_rating` (matchup-specific QB value)

### Multicollinearity

Watch for highly correlated features:
- `nfelo` vs `overall_rating` (r > 0.8 typically)
- `epa_off_L3` vs `epa_off_L5` (r > 0.9)
- `is_home` vs `home_field_advantage` (redundant encoding)

**Solution**: Use Ridge regression (v1.2) or select one representative feature per concept.

---

## Feature Validation Checklist

Before deploying a new feature:

- [ ] Feature is available before kickoff of prediction target
- [ ] Feature does not include current game statistics
- [ ] Feature is documented in this file with tier assignment
- [ ] Feature has been tested for leakage (see `features.validate_no_leakage()`)
- [ ] Feature shows reasonable distribution (no extreme outliers without justification)
- [ ] Feature has non-zero variance (not constant)
- [ ] Feature has logical relationship to spread (positive/negative as expected)

---

## Summary Table

| Tier | Safety Level | Example Features | Use in v1.0 | Use in v1.2 |
|------|--------------|------------------|-------------|-------------|
| **T0** | Always Safe | season, week, is_home, rest_days | Yes | Yes |
| **T1** | Safe (if lagged) | nfelo_diff, epa_off_L3, qb_epa | Yes | Yes |
| **T2** | Caution | vegas_total, projected_spread | No | Conditional |
| **T3** | Experimental | coach_win_pct, weather_severity | No | No |
| **TX** | Forbidden | actual_margin, home_score, post-game stats | Never | Never |

---

**Last Updated**: 2025-11-18
**Maintained By**: Ball Knower Development Team
