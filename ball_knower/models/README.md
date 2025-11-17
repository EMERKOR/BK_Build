# Ball Knower Models

This directory contains all versions of the Ball Knower NFL spread prediction models.

## Model Versions

### v1.1 - Calibrated Spread Model

**File:** `v1_1_calibration.py`

Calibrated version of the deterministic spread model using historical Vegas lines to learn optimal weights.

**Key Features:**
- Uses ordinary least squares (OLS) regression to fit weights
- No sklearn dependency - uses only numpy.linalg.lstsq
- Trains on historical weeks to find optimal component weights
- Provides both calibrated (v1.1) and fixed-weight (v1.0) predictions for comparison

**Model Components:**
- `nfelo_diff`: nfelo rating differential (home - away)
- `substack_power_diff`: Substack overall rating differential
- `epa_off_diff`: Offensive EPA/play differential
- `epa_def_diff`: Defensive EPA/play differential
- `bias`: Learned intercept term

**Functions:**

1. **`prepare_training_matrix(season, weeks, data_dir=None)`**
   - Loads historical weeks and builds training matrix
   - Returns: X (feature matrix), y (Vegas lines), games_df (metadata)

2. **`calibrate_weights(season, weeks, data_dir=None)`**
   - Solves for optimal weights using OLS regression
   - Returns: Dictionary with calibrated weights and bias

3. **`build_week_lines_v1_1(season, week, weights, data_dir=None)`**
   - Generates spread predictions for a given week
   - Returns: DataFrame with v1.1 and v1.0 predictions, edges, and components

**Usage Example:**

```python
from ball_knower.models.v1_1_calibration import calibrate_weights, build_week_lines_v1_1

# Calibrate on historical weeks
weights = calibrate_weights(2025, list(range(1, 11)))

# Generate predictions for current week
lines = build_week_lines_v1_1(2025, 11, weights)
print(lines[['away_team', 'home_team', 'bk_line_v1_1', 'vegas_line', 'edge_v1_1']])
```

**Data Requirements:**

To use calibration, you need historical week data files in `data/current_season/`:
- `power_ratings_nfelo_{season}_week_{week}.csv`
- `epa_tiers_nfelo_{season}_week_{week}.csv`
- `power_ratings_substack_{season}_week_{week}.csv`

Schedule data with Vegas lines:
- `data/cache/schedules_{season}.csv`

**Backtest Notebook:**

See `notebooks/ball_knower_v1_1_backtest.ipynb` for a complete calibration and backtesting workflow including:
- Weight calibration
- Correlation analysis with Vegas
- MAE/RMSE evaluation
- ATS (Against-The-Spread) performance testing

---

## Model Comparison

| Feature | v1.0 (Fixed) | v1.1 (Calibrated) |
|---------|-------------|-------------------|
| Weights | Hand-tuned | Data-driven (OLS) |
| Bias term | None | Learned from data |
| Training | None | Historical weeks |
| Adaptability | Static | Updates with new data |

---

## Next Steps

Future model versions:
- **v1.2**: ML correction layer (Ridge/GBM) on top of v1.1 base predictions
- **v2.0**: Additional features (rest days, weather, QB adjustments, coaching)
