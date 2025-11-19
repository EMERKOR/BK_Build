# Feature Tier System

[Docs Home](README.md) | [Architecture](ARCHITECTURE_OVERVIEW.md) | [Data Sources](DATA_SOURCES.md) | [Feature Tiers](FEATURE_TIERS.md) | [Spec](BALL_KNOWER_SPEC.md) | [Dev Guide](DEVELOPMENT_GUIDE.md)

---

## Overview

This document defines the Ball Knower feature tier system, which categorizes features by their role, reliability, and risk of data leakage. Features are organized into tiers to help developers understand which features are safe to use, which require careful validation, and which are explicitly forbidden.

---

## Tier Definitions

### Tier 0: Structural Features (Always Safe)

**Definition**: Contextual information about the game that is knowable well in advance and carries zero risk of leakage.

**Characteristics**:
- Available before the season starts (or at schedule release)
- Never changes based on game outcomes
- Provides basic temporal and spatial context

**Examples**:
- `season` — NFL season year
- `week` — Week number in season
- `is_home` — Home team indicator (0/1)
- `is_neutral` — Neutral site flag (0/1)
- `rest_days` — Days since last game
- `is_division_game` — Same-division matchup (0/1)
- `is_playoff` — Playoff game flag (0/1)
- `day_of_week` — Game day (0=Thursday, 6=Sunday, etc.)

**Usage Notes**:
- T0 features can be used freely without leakage concerns
- Often interact with other features (e.g., home + rest advantage)
- Provide interpretable baseline effects

---

### Tier 1: Core Model Features (Primary Predictors)

**Definition**: Pre-game team strength metrics that are updated weekly and form the foundation of spread predictions.

**Characteristics**:
- Based on historical performance (past games only)
- Updated before kickoff of prediction target
- No look-ahead bias if properly lagged

**Examples**:
- `nfelo_diff` — nfelo rating differential (home - away)
- `qb_adj_elo` — QB-adjusted Elo rating
- `off_rating` — Offensive power rating
- `def_rating` — Defensive power rating
- `overall_rating` — Overall team rating
- `epa_off_L3` — Offensive EPA per play (last 3 games)
- `epa_def_L3` — Defensive EPA per play (last 3 games)
- `epa_margin_L5` — EPA differential (last 5 games)
- `qb_epa` — QB EPA per play
- `qb_completion_pct` — QB completion percentage

**Critical Requirement**: All rolling features must use `.shift(1)` to exclude current game.

**Example**:
```python
# CORRECT: Excludes current game
df['epa_off_L3'] = df.groupby('team')['epa_off'].shift(1).rolling(3).mean()

# WRONG: Includes current game (LEAKAGE!)
df['epa_off_L3'] = df.groupby('team')['epa_off'].rolling(3).mean()
```

---

### Tier 2: Market Features (Use with Caution)

**Definition**: Betting market information that provides context but must be handled carefully to avoid look-ahead bias.

**Characteristics**:
- Derived from Vegas odds, public betting, or projections
- Can leak future information if not properly time-stamped
- Useful for identifying model disagreement with market

**Examples**:
- `vegas_line_open` — Opening spread
- `vegas_total` — Over/under total
- `public_bet_pct` — Percentage of bets on favorite
- `projected_spread` — Third-party spread projection
- `projected_total` — Third-party total projection

**Critical Rules**:
- NEVER use `vegas_line_close` as a feature when predicting spreads
- It's acceptable as the **target variable** (what we're trying to predict)
- Using it as a feature creates circular logic and perfect backtests

**When to Use**:
- **v1.0**: Do NOT use market features (pure football model)
- **v1.2**: Can use as features IF available before prediction time
- **v2.0**: Can incorporate as meta-features

---

### Tier 3: Experimental Features (Do Not Rely On)

**Definition**: Advanced or novel features that are not yet validated for production use.

**Characteristics**:
- Interesting in theory, unproven in practice
- May require additional data sources not yet integrated
- High risk of overfitting or instability

**Examples**:
- `coach_win_pct` — Head coach career win percentage
- `roster_av` — Team Approximate Value
- `weather_severity` — Wind/temp/precipitation score
- `referee_tendency` — Referee's average penalty differential
- `travel_distance` — Miles traveled for away team
- `primetime_flag` — Thursday/Monday night game indicator

**Recommendation**:
- Include in exploratory analysis
- Do NOT include in production models until validated on holdout data
- Document any experimental features clearly

---

### TX: Forbidden Features (Never Use)

**Definition**: Features that contain post-game information, leaked market data, or obvious leakage.

**Characteristics**:
- Make backtests look amazing but fail in production
- Violate temporal integrity
- Destroy model credibility

**Strictly Prohibited**:
- `actual_margin` — Game result
- `final_score` — Game result
- `home_score` — Home team final score
- `away_score` — Away team final score
- `game_epa` — Calculated from game plays
- `game_total_yards` — Accumulated during game
- `turnovers` — Occurred during game
- `time_of_possession` — Game-specific metric
- `vegas_line_close` (as feature) — Circular prediction target
- `line_from_future_week` — Line from Week N+1
- `epa_off_L3` (without shift) — Includes current game

**Validation Check**:
```python
# Before adding any feature, ask:
# "Could I know this value at 1 PM ET on Sunday, Week N, before any Week N games start?"
# If NO → Do not use it
```

---

## Feature-to-Model Mapping

### Which Features Each Model Version Uses

| Model Version | T0 | T1 | T2 | T3 | TX |
|---------------|----|----|----|----|-----|
| v1.0 | ✅ Yes | ✅ Yes (subset) | ❌ No | ❌ No | ❌ Never |
| v1.2 | ✅ Yes | ✅ Yes (all) | ⚠️ Conditional | ❌ No | ❌ Never |
| v1.3 | ✅ Yes | ✅ Yes (all) | ⚠️ Conditional | ❌ No | ❌ Never |
| v2.0 | ✅ Yes | ✅ Yes (all) | ⚠️ Conditional | 🔬 Testing | ❌ Never |

**Notes**:
- **✅ Yes**: Features are used in production
- **⚠️ Conditional**: Features are used only if available before prediction time
- **🔬 Testing**: Features are being evaluated, not yet in production
- **❌ No/Never**: Features are not used

---

## How Features Flow Through Dataset Builders

### Dataset Builder Workflow

Each model version has a dedicated dataset builder that:

1. **Loads raw data** from `ball_knower/io/loaders/`
2. **Computes features** based on tier requirements
3. **Filters features** to only include allowed tiers
4. **Validates** for leakage and missing values
5. **Exports** to fixtures for testing

**Example**:
```python
# Pseudocode for dataset builder
class DatasetBuilderV10:
    def build(self):
        # Load raw data
        games = load_games()
        ratings = load_ratings()

        # Compute T0 features
        features = compute_structural_features(games)

        # Compute T1 features (subset for v1.0)
        features = compute_core_features(features, ratings)

        # Validate no T2/T3/TX features present
        validate_tiers(features, allowed=[T0, T1])

        # Return dataset
        return features
```

---

## Feature Validation Checklist

Before deploying a new feature:

- [ ] Feature is available before kickoff of prediction target
- [ ] Feature does not include current game statistics
- [ ] Feature is documented in this file with tier assignment
- [ ] Feature has been tested for leakage
- [ ] Feature shows reasonable distribution (no extreme outliers)
- [ ] Feature has non-zero variance (not constant)
- [ ] Feature has logical relationship to spread

---

## References

- [BALL_KNOWER_SPEC.md](BALL_KNOWER_SPEC.md) — Model versions and workflow
- [DATA_SOURCES.md](DATA_SOURCES.md) — Data loading and schemas
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) — Adding new features safely

---

**Status**: This tier system is enforced across all model versions and dataset builders.
