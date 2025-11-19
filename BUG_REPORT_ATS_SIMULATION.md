# Bug Report: ATS Simulation Sign Convention Error

## Summary

The ATS/PnL simulation in `ball_knower/benchmarks/v1_comparison.py` reports impossibly good results (78.5% ATS win rate, 49.9% ROI over 900+ bets) due to **incorrect sign convention in the v1.0 model formula**.

## Root Cause

### The Bug

In `ball_knower/benchmarks/v1_comparison.py` line ~273:

```python
NFELO_COEF = 0.0447
INTERCEPT = 2.67
df['bk_v1_0_spread'] = INTERCEPT + (df['nfelo_diff'] * NFELO_COEF)
```

### The Problem

The v1.0 model is trained to predict Vegas spreads using **betting convention** (negative = home favored), but the hardcoded formula uses the wrong sign.

**Correct formula** (from actual regression):
```python
spread = -1.46 + (-0.042 × nfelo_diff)
# Or equivalently:
spread = -1.46 - (0.042 × nfelo_diff)
```

When `nfelo_diff > 0` (home team stronger):
- Should predict: negative spread (home favored)
- Actually predicts: positive spread (home underdog)

**This inverts every prediction!**

### Evidence

1. **Regression Analysis**
```
Fitted: home_line_close ~ nfelo_diff
Raw coefficient: -0.042048 (NEGATIVE!)
Raw intercept: -1.457171 (NEGATIVE!)
```

2. **Debug Sample**
From `debug_ats_bets.py` output:
```
CHI vs ARI Week 16 2023:
- vegas_line = -4.50 (CHI favored by 4.5)
- bk_spread = 8.16 (wrongly predicts CHI underdog by 8.16!)
- actual_margin = 11.00 (CHI won by 11, easily covered)
- edge = 12.66 (huge because predictions are backwards)
- bet_side = away (bet ARI because model is wrong)
- Result = WIN (ARI covered because CHI was supposed to be favored!)
```

The model thinks CHI is an underdog when they're actually favored, so it bets the opposite side and "wins" when the correct side covers!

3. **Baseline Validation**
```
Always Home: 48.9% win, -6.6% ROI ✓ (expected)
Always Fav:  47.5% win, -9.3% ROI ✓ (expected)
v1.0 Model:  78.5% win, +49.9% ROI ✗ (impossible!)
```

4. **Edge Distribution**
```
Mean edge: 4.72 (systematically positive!)
Mean |edge|: 10.81 (very large)
Home bets: 320
Away bets: 590 (2:1 ratio because predictions inverted)
```

## The Fix

### Option 1: Correct the hardcoded values

```python
NFELO_COEF = -0.0420  # NEGATIVE!
INTERCEPT = -1.46     # NEGATIVE!
df['bk_v1_0_spread'] = INTERCEPT + (df['nfelo_diff'] * NFELO_COEF)
```

### Option 2: Keep positive coef, use minus sign

```python
NFELO_COEF = 0.0420
INTERCEPT = 1.46
df['bk_v1_0_spread'] = -INTERCEPT - (df['nfelo_diff'] * NFELO_COEF)
```

### Option 3: Load from calibration

Don't hardcode at all - compute from actual regression on available data.

## Impact

**All v1.0 comparison results are invalid** due to inverted predictions. The model appears to have 78% ATS win rate when it actually would have ~22% (the inverse).

**v1.2 results may also be affected** - need to verify sign conventions for that model as well.

## Additional Issues

### Potential v1.2 Issue

The v1.2 model (line ~341) loads from JSON:
```python
model_file = config.OUTPUT_DIR / 'ball_knower_v1_2_model.json'
```

JSON shows:
```json
{
  "nfelo_diff": -0.040881578814670826,  // NEGATIVE
  "intercept": -1.837308406904051       // NEGATIVE
}
```

The v1.2 model appears to have correct signs, but should verify the prediction calculation doesn't have similar bugs.

## Verification Tests Needed

1. **Scenario A**: When `model_spread == vegas_line` for all games:
   - ATS win rate should be ~50%
   - ROI should be ~-4.5% (from -110 vig)

2. **Scenario B**: Sign inversion test:
   - If we flip `actual_margin` signs, wins/losses should swap

3. **Scenario C**: Perfect model test:
   - If `model_spread == actual_margin`, betting on all games should have:
   - Win rate = P(actual_margin != vegas_line)
   - Can calculate expected ROI

## Files to Fix

1. `ball_knower/benchmarks/v1_comparison.py`:
   - Fix v1.0 model formula (line ~273)
   - Verify v1.2 model calculation (line ~341+)
   - Add sign convention documentation

2. `tests/test_v1_comparison.py`:
   - Add test for scenario A (model == market → 50% ATS)
   - Add test for scenario B (sign flip → swap wins/losses)
   - Add test comparing to baseline strategies

3. Documentation:
   - Add sign convention guide
   - Document betting convention (negative = favored)
   - Add "trust but verify" note on model performance

## Lessons Learned

1. **Always sanity-check with baselines** - 78% ATS should have been an immediate red flag
2. **Sign conventions are critical** - betting convention (negative = favored) vs. margin convention (positive = home win)
3. **Hardcoded values are dangerous** - should load from source or recompute
4. **Document conventions explicitly** - every variable should state which convention it uses
