# Ball Knower – Feature Tiers Specification

This document defines the feature tier system used to organize inputs to the Ball Knower models.

These tiers do not constrain filenames; they describe logical groupings of features inside the loaded datasets.

---

# 1. Tier Overview

**T0 – Structural Features**
Stadium, surface, altitude, division flags, timezone, game time-of-day.
Low variance, stable, safe for early model versions.

**T1 – Core Football Features**
Team strength indicators, QB/coach ratings, core EPA splits, power rating blends.
These form the backbone of the v1.x predictive model.

**T2 – Market Context Features**
Historical closing line deviations, public bias markers, line movement summaries.
Used for v2.x meta-evaluation and betting layers.

**T3 – Experimental Features**
New or unproven components (advanced player form metrics, compressed stat embeddings).
Not guaranteed to persist and may be removed anytime.

**TX – Forbidden Features**
Anything with leak risk (post-game metrics, drive summaries, play-by-play stats, win probabilities derived from post-hoc models).
Never allowed.

---

# 2. Examples Per Tier (Category-Aware)

## T0 Structural
From `schedule_{season}_week_{week}.csv`:
- neutral_field
- surface_type
- altitude
- outdoor_flag
- game_time
- timezone
- division_matchup
- short_week_flag

## T1 Core Football
From `power_team_*`:
- rating_blend_primary
- rating_blend_secondary
- rating_nfelo
- rating_538_elo

From `epa_team_*`:
- off_epa
- def_epa
- off_success_rate
- def_success_rate

From `qb_ratings_*`:
- qb_tier
- qb_rating
- qb_age
- qb_games_started

From `qb_epa_*`:
- qb_epa_per_play
- qb_epa_total
- qb_epa_vs_avg
- qb_epa_vs_replacement

## T2 Market
Derived from Vegas data (pre-game only):
- closing_spread_prev_week
- early_vs_late_line_movement
- favorite_switch_flag
- public_percent_on_favorite
- book_consensus_line

## T3 Experimental
Internal model extras:
- compressed_off_form
- compressed_def_form
- net_play_efficiency
- pace_index
- injury_severity_score

## TX Forbidden
Any feature containing:
- post-game scores
- drive EPA
- play-by-play aggregates
- live win probability
- after-the-fact injury outcomes
- any stat calculated after kickoff

---

# 3. Responsibilities

- Dataset builders tag features with tiers.
- Models declare which tiers they consume.
- Evaluators must report performance per tier.
- Documentation must be updated whenever a feature moves tiers.

---

End of document.
