# Ball Knower v1.2: Spread Correction Model

## Overview

Ball Knower v1.2 introduces a **machine learning correction layer** that learns residual adjustments on top of the deterministic v1.0 base model. Instead of predicting spreads directly, v1.2 learns what the base model systematically gets wrong and corrects for it.

### Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  v1.2 SPREAD CORRECTION MODEL PIPELINE                     │
└─────────────────────────────────────────────────────────────┘

Input: Canonical pre-game features (team ratings, EPA, etc.)
   │
   ├──► Base Model (v1.0 Deterministic)
   │        └──► Initial spread prediction
   │                │
   │                ├──► Training: Calculate residual = Vegas line - base prediction
   │                │               ML model learns to predict residuals
   │                │
   │                └──► Prediction: Correction = ML model output
   │                                  Final spread = base prediction + correction
   │
   └──► Output: Corrected spread prediction


Training Target: Residuals (Vegas line - base prediction)
NOT the Vegas line directly!

This approach preserves the base model's domain knowledge while allowing
the ML layer to learn systematic biases and market inefficiencies.
```

### Why Residual Learning?

**Traditional Approach (v1.0):**
- Deterministic model: `predicted_spread = f(features)`
- Direct prediction, no learning from market

**v1.2 Residual Approach:**
- Base model: `base_spread = f(features)`
- Residual target: `residual = vegas_line - base_spread`
- ML learns: `correction = g(features, base_spread)`
- Final: `predicted_spread = base_spread + correction`

**Benefits:**
1. **Preserves domain knowledge**: Base model's logic remains intact
2. **Targeted learning**: ML only corrects systematic errors
3. **Stability**: Smaller correction magnitudes = less overfitting risk
4. **Interpretability**: Can analyze what the ML layer is correcting

---

## Features Used

All features are **canonical** (provider-agnostic) and **leakage-free** (pre-game only).

### Core Features (Always Used)

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `base_prediction` | Continuous | v1.0 deterministic model output | Generated from base model |
| `overall_rating_diff` | Continuous | Home - away overall power rating | `ball_knower.io.feature_maps` |
| `epa_margin_diff` | Continuous | Home - away EPA margin (rolling avg) | `ball_knower.io.feature_maps` |
| `offensive_rating_diff` | Continuous | Home - away offensive rating | `ball_knower.io.feature_maps` |
| `defensive_rating_diff` | Continuous | Home - away defensive rating | `ball_knower.io.feature_maps` |

### Optional Features (Used If Available)

| Feature | Type | Description | Availability |
|---------|------|-------------|--------------|
| `qb_adjustment_diff` | Continuous | Home - away QB quality adjustment | Provider: nfelo |
| `is_home` | Binary | 1 = home game, 0 = away/neutral | Always available |
| `div_game` | Binary | 1 = divisional matchup, 0 = other | Requires game metadata |
| `rest_diff` | Continuous | Home - away days of rest advantage | Requires schedule data |

### Critical Constraint: No Vegas Line in Features

**Vegas line is ONLY used as the training target** to calculate residuals. It is **NEVER** included as a feature in the model. This prevents circular logic and leakage.

```python
# ✅ CORRECT (what v1.2 does)
residuals = vegas_lines - base_predictions
correction_model.fit(X_features, residuals)  # X does NOT contain vegas_line

# ❌ WRONG (would be leakage)
X['vegas_line'] = vegas_lines
correction_model.fit(X, spreads)
```

---

## Running the CLI Backtest

### Basic Usage

```bash
python scripts/run_v1_2_correction_backtest.py \
  --season 2024 \
  --train-weeks 1-10 \
  --test-weeks 11-12
```

This will:
1. Load weeks 1-10 from the 2024 season for training
2. Train the correction model on those weeks
3. Evaluate on weeks 11-12
4. Save predictions to `output/v1_2_predictions_2024_weeks_11-12.csv`

### Advanced Options

```bash
# Custom edge thresholds for ATS analysis
python scripts/run_v1_2_correction_backtest.py \
  --season 2024 \
  --train-weeks 1-10 \
  --test-weeks 11-12 \
  --edge-thresholds 0.5,1.0,2.0,3.0,5.0

# Non-consecutive training weeks
python scripts/run_v1_2_correction_backtest.py \
  --season 2024 \
  --train-weeks 1,2,3,5,6,7,9,10 \
  --test-weeks 11

# Adjust regularization strength
python scripts/run_v1_2_correction_backtest.py \
  --season 2024 \
  --train-weeks 1-10 \
  --test-weeks 11-12 \
  --alpha 5.0

# Custom output path
python scripts/run_v1_2_correction_backtest.py \
  --season 2024 \
  --train-weeks 1-10 \
  --test-weeks 11-12 \
  --output custom_predictions.csv

# Quiet mode (suppress verbose output)
python scripts/run_v1_2_correction_backtest.py \
  --season 2024 \
  --train-weeks 1-10 \
  --test-weeks 11-12 \
  --quiet
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--season` | Yes | - | NFL season year (e.g., 2024) |
| `--train-weeks` | Yes | - | Training week range (e.g., "1-10" or "1,2,3,5") |
| `--test-weeks` | Yes | - | Test week range (e.g., "11-12" or "11,12,13") |
| `--edge-thresholds` | No | 0.5,1.0,2.0,3.0 | Edge thresholds for ATS analysis (comma-separated) |
| `--alpha` | No | 10.0 | Ridge regression regularization strength |
| `--output` | No | Auto-generated | Custom output CSV path |
| `--quiet` | No | False | Suppress verbose training/evaluation output |

---

## Output Files

### 1. Predictions CSV

**Filename:** `output/v1_2_predictions_{season}_weeks_{test_weeks}.csv`

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `season` | int | NFL season year |
| `week` | int | Week number |
| `game_id` | string | Unique game identifier |
| `away_team` | string | Away team abbreviation |
| `home_team` | string | Home team abbreviation |
| `vegas_line` | float | Actual Vegas spread line (negative = home favored) |
| `bk_v1_base` | float | Base model (v1.0) prediction |
| `bk_v1_2_corrected` | float | Corrected model (v1.2) prediction |
| `correction` | float | ML correction applied (v1.2 - v1.0) |
| `edge` | float | v1.2 prediction - Vegas line |
| `actual_margin` | float | Actual game margin (if game completed) |

**Example Row:**
```
season,week,game_id,away_team,home_team,vegas_line,bk_v1_base,bk_v1_2_corrected,correction,edge,actual_margin
2024,11,2024_11_KC_BUF,KC,BUF,-2.5,-1.8,-2.3,-0.5,0.2,3
```

**Interpretation:**
- Vegas has Bills favored by 2.5 points (`vegas_line = -2.5`)
- Base model (v1.0) has Bills favored by 1.8 points (`bk_v1_base = -1.8`)
- ML correction adds -0.5 points (`correction = -0.5`)
- v1.2 final prediction: Bills favored by 2.3 points (`bk_v1_2_corrected = -2.3`)
- Edge: v1.2 is 0.2 points higher than Vegas (`edge = 0.2`)
- Bills won by 3 points (`actual_margin = 3`)

### 2. ATS Summary CSV

**Filename:** `output/v1_2_predictions_{season}_weeks_{test_weeks}_ats_summary.csv`

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `edge_threshold` | float | Minimum edge required to make a pick |
| `games` | int | Number of games meeting edge threshold |
| `correct` | int | Number of correct ATS picks |
| `accuracy` | float | Correct / games (as decimal) |
| `roi_estimate` | float | Estimated ROI % assuming -110 odds |

**Example:**
```
edge_threshold,games,correct,accuracy,roi_estimate
0.5,45,26,0.578,5.4
1.0,32,19,0.594,7.0
2.0,18,12,0.667,14.3
3.0,8,6,0.750,22.6
```

**Interpretation:**
- At 0.5-point edge threshold: 45 games, 57.8% accuracy, ~5.4% ROI
- At 2.0-point edge threshold: 18 games, 66.7% accuracy, ~14.3% ROI
- Higher thresholds = fewer games but higher accuracy (typical)

**ROI Calculation:**
- Break-even ATS accuracy at -110 odds: 52.4%
- ROI estimate: `(accuracy - 0.524) * 100`

---

## Programmatic Usage

### Multi-Week Training (Correct Pattern)

The backtest script is the **source of truth** for multi-week training. Here's the correct pattern:

```python
from ball_knower.io import loaders, feature_maps
from ball_knower.models.v1_2_correction import SpreadCorrectionModel
from src.models import DeterministicSpreadModel
from src.nflverse_data import nflverse
from src import config
import pandas as pd

# Initialize base model
base_model = DeterministicSpreadModel(hfa=config.HOME_FIELD_ADVANTAGE)

# Initialize correction model
correction_model = SpreadCorrectionModel(
    base_model=base_model,
    alpha=10.0,
    fit_intercept=True,
    normalize_features=True
)

# Load training data (loop over weeks, NOT load_all_sources(week=range(1,11)))
train_data_list = []
train_weeks = range(1, 11)  # Weeks 1-10

for week in train_weeks:
    # Load game data
    games = nflverse.games(season=2024, week=week)
    games = games[games['spread_line'].notna()].copy()

    if len(games) == 0:
        continue

    # Load ratings via unified loader (SINGLE week at a time)
    all_data = loaders.load_all_sources(season=2024, week=week)

    # Get canonical features
    canonical_ratings = feature_maps.get_canonical_features(all_data['merged_ratings'])

    # Compute feature differentials
    matchups = feature_maps.get_feature_differential(
        canonical_ratings,
        games['home_team'],
        games['away_team'],
        features=['overall_rating', 'epa_margin', 'offensive_rating', 'defensive_rating']
    )

    # Add metadata
    matchups['vegas_line'] = games['spread_line'].values
    matchups['week'] = week

    train_data_list.append(matchups)

# Combine all training weeks
train_data = pd.concat(train_data_list, ignore_index=True)

# Extract Vegas lines as training target
vegas_lines = train_data['vegas_line'].values

# Train correction model
correction_model.fit(train_data, vegas_lines, verbose=True)
```

### Single-Week Prediction

```python
# Load test week data
test_games = nflverse.games(season=2024, week=11)
test_games = test_games[test_games['spread_line'].notna()].copy()

# Load ratings
test_data = loaders.load_all_sources(season=2024, week=11)
canonical_ratings = feature_maps.get_canonical_features(test_data['merged_ratings'])

# Get matchup features
test_matchups = feature_maps.get_feature_differential(
    canonical_ratings,
    test_games['home_team'],
    test_games['away_team'],
    features=['overall_rating', 'epa_margin', 'offensive_rating', 'defensive_rating']
)

# Generate predictions
corrected_spreads = correction_model.predict(test_matchups)

# Compare to Vegas
test_matchups['vegas_line'] = test_games['spread_line'].values
test_matchups['bk_v1_2_prediction'] = corrected_spreads
test_matchups['edge'] = corrected_spreads - test_matchups['vegas_line']
```

### Evaluate Model Performance

```python
# Evaluate on test weeks (with actual game results)
metrics = correction_model.evaluate(
    matchups=test_matchups,
    vegas_lines=test_matchups['vegas_line'].values,
    actual_margins=test_games['actual_margin'].values  # Optional, for ATS
)

print(f"MAE: {metrics['mae']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"Correlation: {metrics['correlation']:.3f}")
print(f"ATS Accuracy: {metrics['ats_accuracy']:.1%}")
```

### Get Feature Importance

```python
# After training, inspect what the ML layer learned
importance = correction_model.get_feature_importance()

for feature, coefficient in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"{feature:30s}: {coefficient:+.4f}")
```

**Example Output:**
```
base_prediction               : +0.8234
epa_margin_diff               : -2.1567
offensive_rating_diff         : +0.4521
defensive_rating_diff         : -0.3892
overall_rating_diff           : +0.0089
```

**Interpretation:**
- Positive coefficient: Feature pushes prediction toward home team
- Negative coefficient: Feature pushes prediction toward away team
- Large magnitude: Strong influence on correction

---

## Model Comparison: v1.0 vs v1.1 vs v1.2

### v1.0: Deterministic Baseline
- **Type:** Deterministic, rule-based
- **Features:** EPA margin, overall rating, offensive/defensive ratings, HFA
- **Weights:** Fixed, hand-tuned
- **Pros:** Simple, interpretable, no overfitting risk
- **Cons:** Cannot learn from market data, systematic biases remain

### v1.1: Enhanced Deterministic (Not Yet Implemented)
- **Type:** Deterministic with structural features
- **Features:** v1.0 features + QB adjustments, rest advantage, recent form
- **Weights:** Fixed, hand-tuned
- **Pros:** More contextual awareness, still interpretable
- **Cons:** Still no learning from Vegas lines

### v1.2: ML Correction Layer (Current)
- **Type:** Hybrid (deterministic base + ML correction)
- **Features:** All canonical pre-game features + base prediction
- **Training:** Ridge regression learns residuals (Vegas - base)
- **Pros:**
  - Learns systematic market patterns
  - Preserves base model domain knowledge
  - Regularized for stability (alpha=10.0)
  - Still uses only pre-game, leakage-free features
- **Cons:**
  - Requires training data (can't predict week 1 without historical data)
  - Black-box correction (less interpretable than v1.0)
  - Risk of overfitting to training weeks

### What's New in v1.2

**Reused from v1.0:**
- Base spread prediction logic (EPA, ratings, HFA)
- Canonical feature mapping layer
- Provider-agnostic architecture
- Leakage-free feature constraints

**New in v1.2:**
- **ML correction layer** (Ridge regression, alpha=10.0)
- **Residual learning** (learns Vegas - base, not spreads directly)
- **Feature normalization** (StandardScaler for stable training)
- **Multi-week training** (accumulates data across week ranges)
- **Backtest infrastructure** (CLI tool, ATS analysis, CSV outputs)
- **Feature importance** (inspect Ridge coefficients)

---

## Model Parameters

### Ridge Regression Hyperparameters

| Parameter | Default | Description | Tuning Guidance |
|-----------|---------|-------------|-----------------|
| `alpha` | 10.0 | L2 regularization strength | Higher = more regularization, less overfitting. Lower = more flexible, risk overfitting. |
| `fit_intercept` | True | Whether to fit intercept term | Keep True unless you have pre-centered features. |
| `normalize_features` | True | Standardize features before fitting | Keep True for stable training with mixed feature scales. |

### When to Adjust Alpha

- **Increase alpha (e.g., 20.0, 50.0)** if:
  - Training MAE << Test MAE (overfitting)
  - Model makes extreme corrections on test weeks
  - Limited training data (< 50 games)

- **Decrease alpha (e.g., 5.0, 1.0)** if:
  - Training MAE ≈ Test MAE (underfitting)
  - Model barely corrects base predictions
  - Large training dataset (> 200 games)

---

## Validation & Testing

### Leakage Checks (All Passing)

✅ **No Vegas line as feature**: Vegas line only used as training target for residuals, never in feature matrix
✅ **All features canonical**: Provider-agnostic via `ball_knower.io.feature_maps`
✅ **All features pre-game**: No play-by-play data, no post-game stats
✅ **Season/week always explicit**: Never pulled from `config.CURRENT_SEASON/CURRENT_WEEK`

### Testing Checklist

Before using v1.2 in production:

- [ ] Run backtest on multiple season/week combinations
- [ ] Verify MAE improvement over v1.0 base model
- [ ] Check ATS accuracy at different edge thresholds
- [ ] Inspect feature importance for sanity (no extreme coefficients)
- [ ] Validate predictions are within reasonable bounds (-20 to +20 points)
- [ ] Ensure no missing feature warnings during training
- [ ] Test with different alpha values to confirm regularization effect

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Cause:** Python environment missing dependencies
**Fix:** Install requirements: `pip install pandas numpy scikit-learn`

### Issue: "No data available" for certain weeks
**Cause:** Data files not present in `data/current_season/` for requested week
**Fix:** Verify files exist with pattern `{category}_{provider}_{season}_week_{week}.csv`

### Issue: "Base model provided but 'base_prediction' not in matchups"
**Cause:** fit() called without generating base predictions first
**Fix:** The model handles this automatically - warning can be ignored if base_model is None

### Issue: Training MAE much lower than test MAE
**Cause:** Overfitting to training weeks
**Fix:** Increase `--alpha` parameter (e.g., 20.0, 50.0) for stronger regularization

### Issue: Corrections are near zero
**Cause:** Model too regularized OR base model already very accurate
**Fix:** Decrease `--alpha` OR verify base model isn't already matching Vegas

---

## Future Enhancements

Potential v1.3+ improvements:

1. **Time-aware features**: Incorporate season trends, week number effects
2. **Market features**: Opening line, line movement (with careful leakage checks)
3. **Ensemble approach**: Combine multiple base models (v1.0 + v1.1) before correction
4. **Non-linear models**: Gradient boosting, neural networks (with strong regularization)
5. **Cross-validation**: Time-series CV for hyperparameter tuning
6. **Weekly retraining**: Update model weekly as season progresses

---

## References

- Base model (v1.0): `src/models.py::DeterministicSpreadModel`
- Correction model (v1.2): `ball_knower/models/v1_2_correction.py::SpreadCorrectionModel`
- Backtest script: `scripts/run_v1_2_correction_backtest.py`
- Canonical features: `ball_knower/io/feature_maps.py::CANONICAL_FEATURE_MAP`
- Unified loader: `ball_knower/io/loaders.py::load_all_sources()`

---

**Version:** 1.2.0
**Last Updated:** 2025-11-17
**Status:** Production-ready, pending backtest validation
