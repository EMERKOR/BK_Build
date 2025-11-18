# Ball Knower – Core Model Specification (v2)
This document defines the purpose, constraints, architecture, and rules of the Ball Knower modeling system.
All modeling work must conform to this specification.

---

# 1. Objective
Ball Knower produces:

1. A football-based predicted point spread (`bk_line`) using only pre-game data.
2. A betting decision layer that evaluates expected value versus the market.
3. A consistent, leak-free historical evaluation against actual game margins and ATS outcomes.

Ball Knower is not designed to predict Vegas spreads; it is designed to predict football and then measure disagreement with the market.

---

# 2. Modeling Constraints

## 2.1 Zero leakage
Only inputs available before kickoff may be used.
No play-by-play, drive results, live stats, or after-the-fact information.

## 2.2 Category-first data naming
All datasets must be stored using the category-first pattern:

```
{category}_{season}week{week}.csv
```

Categories include:
- `epa_team`
- `epa_player`
- `power_team`
- `qb_epa`
- `qb_ratings`
- `schedule`
- `injuries`
- `weather`

See DATA_SOURCES.md for full naming rules.

## 2.3 Compatibility layer
Legacy provider-first filenames may still exist (e.g., `nfelo_power_ratings_2025_week_11.csv`).
The unified loader resolves these automatically but new data must always follow the category-first pattern.

---

# 3. Architecture Overview

## 3.1 Base spread model (v1.x)
Generates `bk_line` using:
- Team power rating differentials
- Home field adjustments
- Structural features (surface, stadium type, altitude, etc.)
- Quarterback and coach ratings
- Team-level EPA and efficiency indicators

## 3.2 Residual correction layer (v1.2+)
Learns small corrections to the base spread using stable, pre-game features.
The correction is applied to the base prediction, not trained to imitate Vegas.

## 3.3 Meta-evaluation layers (v2.x)
Includes:
- Market sentiment signals
- Line movement
- Google Trends
- Public betting bias
- Risk-adjusted edge and bet sizing logic

---

# 4. Data Rules

1. All inputs must originate from category-first files.
2. Each weekly dataset must include exactly one file per category.
3. Missing categories must be surfaced as loader warnings.
4. All files must contain explicit `season` and `week` fields.
5. Merge keys must follow the canonical naming:
   - `game_id`
   - `season`
   - `week`
   - `home_team`
   - `away_team`

---

# 5. Evaluation Rules

## 5.1 Primary evaluation
Ball Knower predictions are compared to:
- Actual game margin
- Vegas closing spread
- ATS outcome

Primary metric: **MAE vs. actual margin**.

## 5.2 Backtesting structure
Models must be trained on past years and tested only on future years using time-aware splits.

---

# 6. Deliverables
A complete Ball Knower model release must include:
- Base spread predictions (`bk_line`)
- Residual-corrected predictions (`bk_line_v1_2` or similar)
- Evaluation reports
- Weekly prediction CSVs
- PredictionTracker compatibility CSVs

---

End of specification.
