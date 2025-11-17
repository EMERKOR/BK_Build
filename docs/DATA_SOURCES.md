# Data Sources (Unified Naming System)

This document defines the naming conventions and structure for all Ball Knower data sources.

---

# 1. Naming Convention

Current-season weekly files must follow:

```
<category>_<source>_<year>_week_<week>.csv
```

Definitions:

- `<category>` = power_ratings, team_epa, qb_metrics, schedule_context
- `<source>` = provider (nfelo, substack, fivethirtyeight, pff, etc.)
- `<year>` = season
- `<week>` = NFL week number

Examples:
- power_ratings_nfelo_2025_week_11.csv
- team_epa_substack_2025_week_11.csv
- qb_metrics_nfelo_2025_week_11.csv
- schedule_context_nfelo_sos_2025_week_11.csv

Blended outputs follow:
- power_ratings_blended_*.csv
- team_epa_blended_*.csv
- qb_metrics_blended_*.csv

---

# 2. Directory Structure

```
data/
  raw/
    current_season/
        power_ratings_*.csv
        team_epa_*.csv
        qb_metrics_*.csv
        schedule_context_*.csv

    historical/
        schedules.parquet
        team_week_epa_2013_2024.csv
        ngs/
        scheme/

    reference/
        nfl_head_coaches.csv
        nfl_AV_data_through_2024.xlsx

  processed/
    current_season/
        power_ratings_blended_*.csv
        team_epa_blended_*.csv
        qb_metrics_blended_*.csv

    modeling/
        team_week_features_2009_2024.parquet
        game_level_model_dataset_2009_2024.parquet
```

---

# 3. Current-Season Categories

## 3.1 Power Ratings (Tier 1)
Files:
```
power_ratings_<source>_<year>_week_<week>.csv
```

Contain:
- overall_rating
- offense_rating
- defense_rating
- qb_adjustment (optional)

Used for Tier 1 strength.

---

## 3.2 Team EPA (Tier 2)
Files:
```
team_epa_<source>_<year>_week_<week>.csv
```

Contain:
- offense_epa_total/pass/rush
- defense_epa_total/pass/rush

Used for rolling form windows.

---

## 3.3 QB Metrics (Tier 1 & 2)
Files:
```
qb_metrics_<source>_<year>_week_<week>.csv
```

Contain:
- qb tier or rating
- qb epa per play
- cpoe
- pressure efficiency

---

## 3.4 Schedule Context / SOS
Files:
```
schedule_context_<source>_<year>_week_<week>.csv
```

Contain:
- sos
- rest days
- travel
- surface
- altitude
- stadium attributes

---

# 4. Historical Data

Stored under `data/raw/historical/`.

Includes:

- schedules.parquet
- team_week_epa_2013_2024.csv
- NGS passing/rushing/receiving
- Scheme/coverage/pressure datasets

---

# 5. Reference Tables

- nfl_head_coaches.csv
- nfl_AV_data_through_2024.xlsx

---

# 6. Processed Modeling Datasets

These are canonical outputs:

- `team_week_features_2009_2024.parquet`
- `game_level_model_dataset_2009_2024.parquet`

Leak-free, versioned, and used for modeling.
