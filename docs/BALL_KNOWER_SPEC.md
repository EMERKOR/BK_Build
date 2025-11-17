# Ball Knower  Core Model Specification (v2)

This document defines the purpose, constraints, architecture, and rules of Ball Knower. All modeling work must conform to this specification.

---

# 1. Objective

Ball Knower is an NFL betting engine designed to produce:

1. A football-based predicted point spread (`bk_line`) using only pre-game information.
2. A betting decision layer that estimates expected value vs the market.
3. A consistent ATS performance evaluation using leak-free historical backtests.

Ball Knower is not a Vegas-line predictor.
The objective is long-run ATS profit, not minimizing MAE vs actual margin.

---

# 2. Data Constraints (No-Leak Rules)

For any game G:

- Features may only use information available before kickoff.
- No post-game or future information is permitted.
- All rolling windows must apply `.shift(1)` at the team-week level.

## Allowed
- Historical games and stats before G.
- Rolling EPA/NGS/team form windows computed with `.shift(1)`.
- Pre-game team ratings (any source).
- QB/coach information known before kickoff.
- Stadium, rest, travel, weather, division context.

## Forbidden
- Any statistic that includes game G or later.
- Ratings updated after game G.
- Season-to-date summaries that are not leak-free.
- Closing line when simulating open-line betting (must choose one explicitly).

All modeling and backtesting must enforce these constraints.

---

# 3. Feature Tiers

Features are organized into conceptual tiers to maintain clarity and interpretability.

## Tier 1  Baseline Strength & Context
- Team power ratings from any provider.
- QB tier metrics.
- Coach tenure/stability.
- Rest days, travel distance, altitude, stadium properties.
- Division game flags.

## Tier 2  Team Form (EPA + NGS)
- Rolling EPA windows (L3, L5, L10).
- Rolling NGS passing metrics.
- Rolling defensive pressure/coverage metrics.
- All computed as team-week ’ merged into game-week as homeaway diffs.

## Tier 3  Scheme & Matchup Profiles
- Offensive scheme tendencies (play-action %, motion %, RPO %, deep shot rate).
- Defensive structure (man %, zone %, single-high %, blitz %, pressure).
- Derived matchup interactions (e.g., blitz-heavy vs QB under pressure).

## Tier 4  Availability / Health (Phase 2)
- QB out, WR1 out, OL injuries.
- Defense starter losses (CB1, EDGE1).
- Injury burden metrics.

Tier 4 will be implemented only when data quality is validated.

---

# 4. Modeling Architecture

Ball Knower consists of two major modeling layers.

---

## Layer 1  Football Line Model

**Goal:** Predict point differential from football information only.

### Target
`margin = home_score - away_score`

### Features
Tier 1 + Tier 2 (+ Tier 3 when ready).

### Rules
- No Vegas line used as a feature.
- Train using chronological splits (older seasons ’ newer).
- Produce:
  - `bk_line_raw`
  - Optional shrunk or uncertainty-calibrated versions.

Model code lives in:
`models/layer1_line/`

---

## Layer 2  Betting Decision Model

**Goal:** Evaluate edge vs market spread and decide whether to bet.

### Inputs:
- `bk_line` (from Layer 1)
- `vegas_line`
- `line_diff = bk_line - vegas_line`
- ATS result flags

### Outputs:
- Probability our implied side covers.
- Expected ATS margin.
- Edge above break-even (52.4% at -110).
- Bet/no-bet decision.

Code lives in:
`models/layer2_betting/`.

---

# 5. Evaluation Metrics

## Layer 1
- MAE, RMSE, bias.
- Calibration buckets (predicted margin ranges vs outcomes).

## Layer 2
- ATS win rate vs 52.4% break-even.
- ROI per unit bet.
- Edge buckets (0.51, 12, 23, 3+).
- Stability across seasons.

---

# 6. Implementation Rules for Assistants

Any assistant (Claude, GPT, etc.) must:

1. Use this spec as the source of truth.
2. Treat all processed datasets as canonical inputs.
3. Avoid Vegas leak in Layer 1.
4. Use chronological splits for training.
5. Add new features in small, interpretable groups.
6. Document all new model files with:
   - Data ranges
   - Features used
   - Evaluation results

Experimental or speculative code must go into `archive/`.
