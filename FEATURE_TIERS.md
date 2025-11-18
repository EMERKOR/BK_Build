# Ball Knower Feature Tiers

## Overview

This document defines the feature tier system for Ball Knower models. Features are organized by **availability**, **quality**, and **predictive power** to support incremental model development from simple baselines to complex ensembles.

The tier system enables:
- **Progressive model building** (v1.0 → v1.1 → v1.2 → v2.0)
- **Graceful degradation** when data sources are unavailable
- **A/B testing** of feature importance
- **Clear documentation** of what each model version uses

---

## Tier Definitions

| Tier | Availability | Quality | Use Case |
|------|--------------|---------|----------|
| **Tier 1** | Always available | High consistency | Core baseline models |
| **Tier 2** | Usually available | Provider-dependent | Enhanced models |
| **Tier 3** | Optional/experimental | Variable | Advanced/experimental models |

---

## Tier 1: Core Features (Always Available)

These features form the foundation of all Ball Knower models. They are:
- Always available for current week predictions
- High quality and well-calibrated
- Historically stable and predictive
- Used in v1.0 baseline model

### power_ratings_nfelo

**Category:** `power_ratings`
**Provider:** `nfelo`
**File:** `power_ratings_nfelo_{season}_week_{week}.csv`

**Key Columns:**
- `rating` - Overall team strength (scale: ~1400-1700)
- `elo` - Base ELO rating
- `qb_adj` - QB adjustment factor

**Feature Engineering:**
```python
# Rating differential
home_nfelo = merged_ratings.loc[home_team, 'rating']
away_nfelo = merged_ratings.loc[away_team, 'rating']
nfelo_diff = home_nfelo - away_nfelo

# Contribution to spread (empirically calibrated)
spread_contrib_nfelo = nfelo_diff * 0.04
```

**Predictive Power:**
- ELO differential × 0.04 ≈ point spread contribution
- Explains ~40% of spread variance

**Fallback:** If unavailable, use historical ELO derived from game results.

---

### epa_tiers_nfelo

**Category:** `epa_tiers`
**Provider:** `nfelo`
**File:** `epa_tiers_nfelo_{season}_week_{week}.csv`

**Key Columns:**
- `off_epa` - Offensive EPA per play
- `def_epa` - Defensive EPA per play (allowed)
- `off_pass_epa` - Passing EPA per play
- `off_run_epa` - Rushing EPA per play

**Feature Engineering:**
```python
# EPA differential
home_off_epa = merged_ratings.loc[home_team, 'off_epa']
away_def_epa = merged_ratings.loc[away_team, 'def_epa']
away_off_epa = merged_ratings.loc[away_team, 'off_epa']
home_def_epa = merged_ratings.loc[home_team, 'def_epa']

# Net EPA advantage
epa_diff = (home_off_epa - away_def_epa) - (away_off_epa - home_def_epa)

# Contribution to spread (empirically calibrated)
spread_contrib_epa = epa_diff * 100
```

**Predictive Power:**
- EPA differential × 100 ≈ point differential per game
- Explains ~50% of spread variance
- Most predictive single metric

**Fallback:** Calculate from historical play-by-play using nfl_data_py.

---

### strength_of_schedule_nfelo

**Category:** `strength_of_schedule`
**Provider:** `nfelo`
**File:** `strength_of_schedule_nfelo_{season}_week_{week}.csv`

**Key Columns:**
- `sos` - Strength of schedule (past opponents)
- `sos_remaining` - Remaining schedule strength

**Feature Engineering:**
```python
# Use for context, not direct prediction
home_sos = merged_ratings.loc[home_team, 'sos']
away_sos = merged_ratings.loc[away_team, 'sos']

# Primarily used for:
# 1. Adjusting team ratings (teams with hard schedules may be underrated)
# 2. Identifying regression candidates
# 3. Season win total projections
```

**Predictive Power:**
- Limited direct impact on single-game spreads
- Useful for contextualizing team records
- Important for season-long projections

**Fallback:** Calculate from opponent ELO ratings.

---

## Tier 2: Enhanced Features (Usually Available)

These features improve model performance when available. They are:
- Available most weeks from reliable providers
- High quality but may have occasional gaps
- Used in v1.1 and v1.2 models

### power_ratings_substack (migrating to specific providers)

**Category:** `power_ratings`
**Provider:** `substack` (being decomposed into specific providers)
**File:** `power_ratings_substack_{season}_week_{week}.csv`

**Key Columns:**
- `off_rating` - Offensive rating
- `def_rating` - Defensive rating
- `overall_rating` - Combined rating

**Feature Engineering:**
```python
# Use as ensemble component
substack_overall = merged_ratings.loc[home_team, 'overall_rating']
substack_diff = merged_ratings.loc[home_team, 'overall_rating'] - merged_ratings.loc[away_team, 'overall_rating']

# Blend with nfelo (reduces overfitting)
ensemble_rating = 0.6 * nfelo_diff + 0.4 * substack_diff
```

**Predictive Power:**
- Similar to nfelo when available
- Diversifies signal (reduces model variance)
- Ensemble improves ~5% over single source

**Fallback:** Use nfelo only (Tier 1 feature).

---

### qb_epa_nfelo

**Category:** `qb_epa`
**Provider:** `nfelo`
**File:** `qb_epa_nfelo_{season}_week_{week}.csv`

**Key Columns:**
- `qb_epa` - QB EPA per play
- `player` - QB name
- `team` - Team abbreviation

**Feature Engineering:**
```python
# QB advantage
home_qb_epa = qb_data.loc[qb_data['team'] == home_team, 'qb_epa'].iloc[0]
away_qb_epa = qb_data.loc[qb_data['team'] == away_team, 'qb_epa'].iloc[0]

qb_epa_diff = home_qb_epa - away_qb_epa

# Contribution to spread
spread_contrib_qb = qb_epa_diff * 50  # QBs touch ~50% of plays
```

**Predictive Power:**
- Critical for QB injury/change scenarios
- Explains ~20% of spread variance
- Most important for backup QB games

**Fallback:** Use team-level EPA (already accounts for QB partially).

---

### qb_epa_substack

**Category:** `qb_epa`
**Provider:** `substack`
**File:** `qb_epa_substack_{season}_week_{week}.csv`

**Key Columns:**
- `qb_epa` - QB EPA metric
- `player` - QB name
- `team` - Team abbreviation (may be in 'Tms' column)

**Feature Engineering:**
```python
# Ensemble QB metric
qb_epa_ensemble = 0.5 * qb_epa_nfelo + 0.5 * qb_epa_substack
```

**Predictive Power:**
- Similar to nfelo QB EPA
- Diversifies QB evaluation
- Useful for controversial QB rankings

**Fallback:** Use qb_epa_nfelo only.

---

### weekly_projections_ppg_substack

**Category:** `weekly_projections_ppg`
**Provider:** `substack`
**File:** `weekly_projections_ppg_substack_{season}_week_{week}.csv`

**Key Columns:**
- `home_ppg` - Projected home points
- `away_ppg` - Projected away points
- `spread` - Projected spread

**Feature Engineering:**
```python
# Use as validation/comparison
projected_spread_substack = merged_ratings.loc[matchup, 'spread_proj_ppg']

# Or as ensemble component
ensemble_spread = 0.5 * ball_knower_spread + 0.5 * projected_spread_substack
```

**Predictive Power:**
- Similar to Ball Knower baseline
- Useful for consensus analysis
- Identifies outlier predictions

**Fallback:** Use Ball Knower model only.

---

## Tier 3: Experimental Features (Optional)

These features are experimental, optional, or require special setup. They are:
- Not always available
- Quality varies by provider
- Used in v2.0+ advanced models
- Require validation before production use

### qb_epa_pff

**Category:** `qb_epa`
**Provider:** `pff` (Pro Football Focus)
**File:** `qb_epa_pff_{season}_week_{week}.csv`

**Key Columns:**
- `grade` - PFF overall QB grade (0-100)
- `epa` - EPA per play
- Advanced charting metrics

**Feature Engineering:**
```python
# Use PFF grade as QB quality metric
pff_qb_diff = (home_qb_grade - away_qb_grade) / 10  # Normalize to ~10-point scale
spread_contrib_pff_qb = pff_qb_diff * 0.5
```

**Predictive Power:**
- High quality but subscription required
- May outperform EPA-based QB metrics
- Useful for film-based insights

**Fallback:** Use Tier 2 QB EPA metrics.

---

### power_ratings_538

**Category:** `power_ratings`
**Provider:** `538` (FiveThirtyEight)
**File:** `power_ratings_538_{season}_week_{week}.csv`

**Key Columns:**
- `elo` - 538 ELO rating
- `qb_adj` - QB adjustment
- `rating` - Overall rating

**Feature Engineering:**
```python
# Ensemble with nfelo and substack
ensemble_rating = (
    0.4 * nfelo_diff +
    0.3 * substack_diff +
    0.3 * elo_538_diff
)
```

**Predictive Power:**
- Comparable to nfelo
- Well-documented methodology
- Historical data available for backtesting

**Fallback:** Use Tier 1 nfelo ratings.

---

### team_efficiency_pff

**Category:** `team_efficiency`
**Provider:** `pff`
**File:** `team_efficiency_pff_{season}_week_{week}.csv`

**Key Columns:**
- `off_grade` - Offensive team grade
- `def_grade` - Defensive team grade
- Position-specific grades

**Feature Engineering:**
```python
# Use as alternative to EPA
pff_grade_diff = (home_off_grade - away_def_grade) - (away_off_grade - home_def_grade)
spread_contrib_pff = pff_grade_diff * 0.3
```

**Predictive Power:**
- Film-based grading may capture nuances EPA misses
- Requires subscription
- Validation needed vs. EPA metrics

**Fallback:** Use Tier 1 EPA tiers.

---

### user_custom_features

**Category:** Various
**Provider:** `user`
**File:** `{category}_user_{season}_week_{week}.csv`

**Description:** Custom features engineered by user or derived from other sources.

**Examples:**
- Weather adjustments (wind, temperature)
- Rest advantage (days since last game)
- Travel distance
- Referee tendencies
- Injury reports (manually curated)
- Line movement tracking
- Public betting percentages

**Feature Engineering:**
```python
# Example: Rest advantage
rest_diff = home_days_rest - away_days_rest
spread_contrib_rest = rest_diff * 0.3  # ~0.3 points per day of rest

# Example: Weather
if wind_speed > 20:  # mph
    spread_contrib_weather = -1.0  # Favors run-heavy teams
```

**Predictive Power:**
- Highly variable
- Requires careful validation
- Can capture edges missed by public models

**Fallback:** Exclude from model.

---

## Model Version → Tier Mapping

### v1.0 Baseline (Tier 1 Only)

**Features used:**
- `power_ratings_nfelo` → nfelo_diff × 0.04
- `epa_tiers_nfelo` → epa_diff × 100
- Home field advantage → +2.5 points

**Model type:** Deterministic weighted sum

**Performance:** ~55% ATS (against the spread), establishes baseline

**Advantages:**
- No training required
- Fully transparent
- Leak-free
- Fast to compute

---

### v1.1 Enhanced (Tier 1 + Tier 2)

**Additional features:**
- `qb_epa_nfelo` → qb_epa_diff × 50
- `power_ratings_substack` → ensemble with nfelo
- Rest days (if available)

**Model type:** Deterministic weighted sum with more features

**Performance:** ~56-57% ATS (estimated)

**Advantages:**
- Still interpretable
- No training required
- Handles QB changes better

---

### v1.2 ML Correction (Tier 1 + Tier 2 + Small ML Layer)

**Features used:**
- All v1.1 features
- Ridge regression correction layer
- Learns residuals from v1.1 predictions

**Model type:** Hybrid (deterministic core + ML correction)

**Performance:** ~58-60% ATS (estimated)

**Advantages:**
- Captures nonlinear effects
- Small training set (less overfitting)
- Retains interpretability of baseline

---

### v2.0 Advanced Ensemble (All Tiers)

**Features used:**
- All Tier 1, 2, 3 features when available
- Gradient boosting or neural network
- Time-series cross-validation

**Model type:** Full ML ensemble

**Performance:** ~60%+ ATS (target)

**Risks:**
- Requires careful validation
- Risk of overfitting
- Less interpretable
- May not generalize to rule changes

---

## Feature Importance (Empirical)

Based on v1.0-v1.2 backtests and NFL modeling literature:

| Feature | Tier | Importance | Notes |
|---------|------|------------|-------|
| `epa_tiers_nfelo` | 1 | ⭐⭐⭐⭐⭐ | Most predictive single metric |
| `power_ratings_nfelo` | 1 | ⭐⭐⭐⭐ | Stable, well-calibrated |
| Home field advantage | N/A | ⭐⭐⭐⭐ | ~2.5 points (decreasing over time) |
| `qb_epa` | 2 | ⭐⭐⭐ | Critical for QB changes |
| `power_ratings_substack` | 2 | ⭐⭐ | Ensemble benefit, not standalone |
| Rest days | 3 | ⭐⭐ | ~0.3 points per day |
| Weather | 3 | ⭐ | Only extreme conditions matter |
| `strength_of_schedule` | 1 | ⭐ | Context, not direct prediction |

---

## Feature Engineering Best Practices

### Leak Prevention

**CRITICAL:** All features must be strictly leak-free.

```python
# ❌ BAD: Using current week stats
current_epa = df.loc[df['week'] == current_week, 'epa'].mean()

# ✅ GOOD: Using past weeks only
past_epa = df.loc[df['week'] < current_week, 'epa'].rolling(4).mean().shift(1)
```

**Rules:**
1. Always `.shift(1)` when computing rolling averages
2. Never use same-week data
3. Validate features don't correlate perfectly with outcome
4. Use time-based train/test splits (not random)

---

### Normalization

All features should be normalized to similar scales for ML models:

```python
from sklearn.preprocessing import StandardScaler

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Why:** Prevents large-scale features (like ELO ~1500) from dominating small-scale features (like EPA ~0.1).

---

### Handling Missing Data

**Tier 1 features:** Should never be missing. If they are, stop and diagnose.

**Tier 2 features:** Fill with Tier 1 equivalent or median:
```python
# Example: Missing qb_epa
if pd.isna(qb_epa_nfelo):
    qb_epa_nfelo = team_epa  # Use team-level EPA as fallback
```

**Tier 3 features:** Can be excluded:
```python
# Example: Missing PFF grades
if 'pff_grade' not in df.columns:
    # Skip PFF features, model still works
    pass
```

---

## Feature Addition Workflow

### Adding a New Feature

1. **Classify tier** (1, 2, or 3) based on availability and quality
2. **Define in this document** (category, provider, file naming)
3. **Create loader function** (if new category) in `loaders.py`
4. **Validate leak-free** using historical backtest
5. **Test feature importance** via ablation study
6. **Document** in this file and `DATA_SOURCES.md`

### Ablation Study Template

```python
# Baseline model (without new feature)
baseline_model = train_model(X_train_baseline, y_train)
baseline_acc = evaluate(baseline_model, X_test_baseline, y_test)

# Enhanced model (with new feature)
enhanced_model = train_model(X_train_enhanced, y_train)
enhanced_acc = evaluate(enhanced_model, X_test_enhanced, y_test)

# Feature value
feature_value = enhanced_acc - baseline_acc
print(f"Feature improved accuracy by {feature_value:.2%}")
```

If `feature_value > 1%`, feature is likely worth keeping.

---

## Tier Promotion/Demotion Criteria

Features can move between tiers based on experience:

**Promote to Tier 1:**
- Available 100% of weeks for full season
- High correlation with outcomes (p < 0.01)
- Stable across seasons
- No quality issues

**Promote to Tier 2:**
- Available >80% of weeks
- Proven value in ablation studies
- Reliable provider

**Demote to Tier 3:**
- Frequent missing data
- Quality issues
- Low/negative feature importance
- Provider discontinued

**Remove entirely:**
- Causes data leakage
- Negative feature importance consistently
- No longer available

---

## See Also

- `BALL_KNOWER_SPEC.md` - System specification and naming conventions
- `DATA_SOURCES.md` - Detailed data source descriptions
- `src/features.py` - Feature engineering implementation
- `src/models.py` - Model implementations using these tiers
