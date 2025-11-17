# Feature Tiers (Unified Naming System)

This document maps concrete data columns to the four Ball Knower feature tiers.

---

# Tier 1  Baseline Strength & Context

## Inputs (team-week level)
- power_ratings_* files
- qb_metrics_* files (tiers)
- coach tenure (from reference tables)
- rest_days
- travel_distance
- stadium_roof, stadium_surface, altitude
- division game flags

## Derived Game-Level
- rating_diff = home - away (for any rating source)
- qb_tier_diff
- rest_diff
- travel_diff
- altitude_home flag

---

# Tier 2  Team Form (EPA + NGS)

## Inputs (team-week level)
From team_epa_* and NGS data:
- offense_epa_total/pass/rush
- defense_epa_total/pass/rush
- cpoe, time_to_throw, separation, pressure metrics

Apply `.shift(1)` and rolling windows:
- *_L3
- *_L5
- *_L10

## Derived Game-Level
- off_form_diff_L5
- def_form_diff_L5
- pass_form_diff_L5
- rush_form_diff_L5
- qb_cpoe_diff_L3

Form features should be compressed into meaningful aggregates.

---

# Tier 3  Scheme & Matchup Profiles

## Inputs
Scheme/snap charting from historical/scheme data:
- coverage distributions (man %, zone %)
- single-high %, two-high %
- blitz %, pressure %
- box counts
- offensive: play-action %, RPO %, motion %, deep shot %, shotgun %

## Derived Matchup Features
Examples:
- blitz_vs_qb_under_pressure
- man_vs_man_sensitive
- two_high_vs_run_heavy

Tier 3 = style vs style, not raw stats.

---

# Tier 4  Availability / Health (Phase 2)

## Inputs
Injury reports or snap count deltas:
- qb_out_flag
- wr1_out_flag
- ol_injured_starters
- def_injured_starters

## Derived
- injury_burden_score

Will be added once data reliability is validated.
