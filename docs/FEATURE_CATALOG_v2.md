# Ball_Knower v2.0 Feature Catalog

**Auto-generated feature catalog and leakage risk analysis**

---

## 1. Summary

- **Total unique columns analyzed:** 207
- **Total CSV files scanned:** 14
- **Data directory:** `data/`

### Columns by Role

- **meta_misc:** 118
- **pre_game_structure:** 20
- **id_key:** 19
- **pre_game_market:** 17
- **in_game_stats:** 15
- **pre_game_team_strength:** 14
- **target_outcome:** 4

### Columns by Leakage Risk

- **unknown:** 118
- **low:** 70
- **medium:** 15
- **high:** 4

---

## 2. Columns by Role

### id_key (19 columns)

| Column Name | Leakage Risk | Source Files | Notes |
|-------------|--------------|--------------|-------|
| `Change (Week)` | low | current_season/nfelo_qb_rankings_2025_week_11.csv | Identified as key/identifier column |
| `Change (Year)` | low | current_season/nfelo_qb_rankings_2025_week_11.csv | Identified as key/identifier column |
| `Date` | low | current_season/substack_weekly_proj_elo_2025_week_11.csv (+1 more) | Identified as key/identifier column |
| `Season` | low | current_season/epa_tiers_nfelo_2025_week_11.csv (+5 more) | Identified as key/identifier column |
| `Seasons` | low | reference/nfl_head_coaches.csv | Identified as key/identifier column |
| `Team` | low | current_season/epa_tiers_nfelo_2025_week_11.csv (+3 more) | Identified as key/identifier column |
| `Team.1` | low | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Identified as key/identifier column |
| `away_qb_id` | low | cache/schedules_2025.csv | Identified as key/identifier column |
| `away_team` | low | cache/schedules_2025.csv | Identified as key/identifier column |
| `game_id` | low | cache/schedules_2025.csv | Identified as key/identifier column |
| `home_qb_id` | low | cache/schedules_2025.csv | Identified as key/identifier column |
| `home_team` | low | cache/schedules_2025.csv | Identified as key/identifier column |
| `nfl_detail_id` | low | cache/schedules_2025.csv | Identified as key/identifier column |
| `old_game_id` | low | cache/schedules_2025.csv | Identified as key/identifier column |
| `season` | low | cache/schedules_2025.csv (+1 more) | Identified as key/identifier column |
| `stadium_id` | low | cache/schedules_2025.csv | Identified as key/identifier column |
| `team` | low | team_week_epa_2013_2024.csv | Identified as key/identifier column |
| `week` | low | cache/schedules_2025.csv (+1 more) | Identified as key/identifier column |
| `weekday` | low | cache/schedules_2025.csv | Identified as key/identifier column |

### in_game_stats (15 columns)

| Column Name | Leakage Risk | Source Files | Notes |
|-------------|--------------|--------------|-------|
| `Air Yards` | medium | current_season/nfelo_qb_rankings_2025_week_11.csv | In-game stat - verify timing before use |
| `EPA/Play` | medium | current_season/epa_tiers_nfelo_2025_week_11.csv | In-game stat - verify timing before use |
| `EPA/Play Against` | medium | current_season/epa_tiers_nfelo_2025_week_11.csv | In-game stat - verify timing before use |
| `Sacks` | medium | current_season/nfelo_qb_rankings_2025_week_11.csv | In-game stat - verify timing before use |
| `Sacks.1` | medium | current_season/nfelo_qb_rankings_2025_week_11.csv | In-game stat - verify timing before use |
| `Yards` | medium | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv (+1 more) | In-game stat - verify timing before use |
| `Yards.1` | medium | current_season/nfelo_qb_rankings_2025_week_11.csv | In-game stat - verify timing before use |
| `def_epa_per_play` | medium | team_week_epa_2013_2024.csv | In-game stat - verify timing before use |
| `def_plays` | medium | team_week_epa_2013_2024.csv | In-game stat - verify timing before use |
| `def_success_rate` | medium | team_week_epa_2013_2024.csv | In-game stat - verify timing before use |
| `off_epa_per_play` | medium | team_week_epa_2013_2024.csv | In-game stat - verify timing before use |
| `off_pass_plays` | medium | team_week_epa_2013_2024.csv | In-game stat - verify timing before use |
| `off_plays` | medium | team_week_epa_2013_2024.csv | In-game stat - verify timing before use |
| `off_rush_plays` | medium | team_week_epa_2013_2024.csv | In-game stat - verify timing before use |
| `off_success_rate` | medium | team_week_epa_2013_2024.csv | In-game stat - verify timing before use |

### meta_misc (118 columns)

| Column Name | Leakage Risk | Source Files | Notes |
|-------------|--------------|--------------|-------|
| `/ Catch` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `/ DB` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `/ Game` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `/ Route` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `/ Target` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `1st Downs` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `3rd Down` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `ANY/A` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `ATS %` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Actual` | unknown | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Could not classify - manual review needed |
| `Against` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `All` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `Atts` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Avg PA` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Avg PF` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `CB` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `CB Opps` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `CB%` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `CPOE` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `CROE` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `Carries` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Catch Rate` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `Coach` | unknown | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv (+1 more) | Could not classify - manual review needed |
| `Comp` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Comp%` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Conf Champ` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Dif` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `Dog` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Favorite` | unknown | current_season/substack_weekly_proj_elo_2025_week_11.csv (+2 more) | Could not classify - manual review needed |
| `Film` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `For` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `GP` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `Games` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Hold` | unknown | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Could not classify - manual review needed |
| `INT%` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `INTs` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `INTs.1` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Inc` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `L` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Losses` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Matchup` | unknown | current_season/substack_weekly_proj_elo_2025_week_11.csv (+1 more) | Could not classify - manual review needed |
| `Opponent` | unknown | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Could not classify - manual review needed |
| `Over` | unknown | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Could not classify - manual review needed |
| `Over.1` | unknown | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Could not classify - manual review needed |
| `Pass` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `Pass.1` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `Passer Rtg` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Penalties` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Play` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `Play.1` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `Play.2` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `Player` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `Points` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Pythag` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `QB` | unknown | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv (+1 more) | Could not classify - manual review needed |
| `QB Adj` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `Recs` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `Red Zone` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `Return (%)` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Return (units)` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Rush` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `Rush.1` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `Rushing` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Starts` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Status` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Success` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Super Bowls` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `T` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `TD%` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `TD%-INT%` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `TDs` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv (+1 more) | Could not classify - manual review needed |
| `TDs.1` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Target Rate` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `Targets` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `Ties` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Time (ET)` | unknown | current_season/substack_weekly_proj_elo_2025_week_11.csv (+1 more) | Could not classify - manual review needed |
| `Under` | unknown | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Could not classify - manual review needed |
| `Under.1` | unknown | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Could not classify - manual review needed |
| `W` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `WPA / DB` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `Win %` | unknown | reference/nfl_head_coaches.csv | Could not classify - manual review needed |
| `Win Prob.` | unknown | current_season/substack_weekly_proj_elo_2025_week_11.csv (+1 more) | Could not classify - manual review needed |
| `Wins` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv (+1 more) | Could not classify - manual review needed |
| `WoW` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `X.1` | unknown | current_season/power_ratings_substack_2025_week_11.csv | Could not classify - manual review needed |
| `X.2` | unknown | current_season/power_ratings_substack_2025_week_11.csv (+1 more) | Could not classify - manual review needed |
| `X.20` | unknown | current_season/qb_epa_substack_2025_week_11.csv | Could not classify - manual review needed |
| `X.23` | unknown | current_season/qb_epa_substack_2025_week_11.csv | Could not classify - manual review needed |
| `X.25` | unknown | current_season/qb_epa_substack_2025_week_11.csv | Could not classify - manual review needed |
| `X.26` | unknown | current_season/qb_epa_substack_2025_week_11.csv | Could not classify - manual review needed |
| `X.3` | unknown | current_season/power_ratings_substack_2025_week_11.csv | Could not classify - manual review needed |
| `X.4` | unknown | current_season/power_ratings_substack_2025_week_11.csv (+1 more) | Could not classify - manual review needed |
| `X.5` | unknown | current_season/power_ratings_substack_2025_week_11.csv | Could not classify - manual review needed |
| `X.6` | unknown | current_season/power_ratings_substack_2025_week_11.csv (+1 more) | Could not classify - manual review needed |
| `X.7` | unknown | current_season/power_ratings_substack_2025_week_11.csv | Could not classify - manual review needed |
| `X.8` | unknown | current_season/power_ratings_substack_2025_week_11.csv | Could not classify - manual review needed |
| `X.9` | unknown | current_season/qb_epa_substack_2025_week_11.csv | Could not classify - manual review needed |
| `YAC` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `YPA` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `YPC` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv (+1 more) | Could not classify - manual review needed |
| `YPG` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv | Could not classify - manual review needed |
| `YTD` | unknown | current_season/power_ratings_nfelo_2025_week_11.csv | Could not classify - manual review needed |
| `aDOT` | unknown | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv (+1 more) | Could not classify - manual review needed |
| `div_game` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `espn` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `ftn` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `game_type` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `gameday` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `gametime` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `gsis` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `overtime` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `pff` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `pfr` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `referee` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `temp` | unknown | cache/schedules_2025.csv | Could not classify - manual review needed |
| `vs Sticks` | unknown | current_season/nfelo_qb_rankings_2025_week_11.csv | Could not classify - manual review needed |
| `~~~2025 Forecast~~~` | unknown | current_season/power_ratings_substack_2025_week_11.csv | Could not classify - manual review needed |
| `~~~Per 17 Gms~~~` | unknown | current_season/qb_epa_substack_2025_week_11.csv | Could not classify - manual review needed |

### pre_game_market (17 columns)

| Column Name | Leakage Risk | Source Files | Notes |
|-------------|--------------|--------------|-------|
| `Adj. Total` | low | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Market line/odds - available pre-game |
| `Avg Spread` | low | reference/nfl_head_coaches.csv | Market line/odds - available pre-game |
| `Total` | low | current_season/nfelo_nfl_receiving_leaders_2025_week_11.csv (+1 more) | Market line/odds - available pre-game |
| `Total WPA` | low | current_season/nfelo_qb_rankings_2025_week_11.csv | Market line/odds - available pre-game |
| `Vegas Total` | low | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Market line/odds - available pre-game |
| `Win Total` | low | current_season/strength_of_schedule_nfelo_2025_week_11.csv | Market line/odds - available pre-game |
| `away_moneyline` | low | cache/schedules_2025.csv | Market line/odds - available pre-game |
| `away_spread_odds` | low | cache/schedules_2025.csv | Market line/odds - available pre-game |
| `def_epa_total` | low | team_week_epa_2013_2024.csv | Market line/odds - available pre-game |
| `home_moneyline` | low | cache/schedules_2025.csv | Market line/odds - available pre-game |
| `home_spread_odds` | low | cache/schedules_2025.csv | Market line/odds - available pre-game |
| `off_epa_total` | low | team_week_epa_2013_2024.csv | Market line/odds - available pre-game |
| `over_odds` | low | cache/schedules_2025.csv | Market line/odds - available pre-game |
| `spread_line` | low | cache/schedules_2025.csv | Market line/odds - available pre-game |
| `total` | low | cache/schedules_2025.csv | Market line/odds - available pre-game |
| `total_line` | low | cache/schedules_2025.csv | Market line/odds - available pre-game |
| `under_odds` | low | cache/schedules_2025.csv | Market line/odds - available pre-game |

### pre_game_structure (20 columns)

| Column Name | Leakage Risk | Source Files | Notes |
|-------------|--------------|--------------|-------|
| `Away` | low | reference/nfl_head_coaches.csv | Structural/environmental factor - known pre-game |
| `Division` | low | reference/nfl_head_coaches.csv | Structural/environmental factor - known pre-game |
| `Home` | low | reference/nfl_head_coaches.csv | Structural/environmental factor - known pre-game |
| `Non Division` | low | reference/nfl_head_coaches.csv | Structural/environmental factor - known pre-game |
| `Off a Bye` | low | reference/nfl_head_coaches.csv | Structural/environmental factor - known pre-game |
| `Playoff Berths` | low | reference/nfl_head_coaches.csv | Structural/environmental factor - known pre-game |
| `Playoffs` | low | reference/nfl_head_coaches.csv | Structural/environmental factor - known pre-game |
| `away_coach` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `away_qb_name` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `away_rest` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `away_score` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `home_coach` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `home_qb_name` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `home_rest` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `home_score` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `location` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `roof` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `stadium` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `surface` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |
| `wind` | low | cache/schedules_2025.csv | Structural/environmental factor - known pre-game |

### pre_game_team_strength (14 columns)

| Column Name | Leakage Risk | Source Files | Notes |
|-------------|--------------|--------------|-------|
| `Avg. Opp. Rating` | low | current_season/strength_of_schedule_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `Avg. Opp. Rating.1` | low | current_season/strength_of_schedule_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `Avg. Opp. Rating.2` | low | current_season/strength_of_schedule_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `Current Rating` | low | current_season/strength_of_schedule_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `Elo` | low | current_season/power_ratings_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `Original Rating` | low | current_season/strength_of_schedule_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `QB Elo` | low | current_season/nfelo_qb_rankings_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `QBR` | low | current_season/nfelo_qb_rankings_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `Rank` | low | current_season/strength_of_schedule_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `Rank.1` | low | current_season/strength_of_schedule_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `Rank.2` | low | current_season/strength_of_schedule_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `Value` | low | current_season/power_ratings_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `nfelo` | low | current_season/power_ratings_nfelo_2025_week_11.csv | Team/player strength metric - pre-game rating |
| `~~~Ratings~~~` | low | current_season/power_ratings_substack_2025_week_11.csv | Team/player strength metric - pre-game rating |

### target_outcome (4 columns)

| Column Name | Leakage Risk | Source Files | Notes |
|-------------|--------------|--------------|-------|
| `Avg Margin` | high | reference/nfl_head_coaches.csv | Game outcome/result - prediction target |
| `Result` | high | current_season/nfelo_nfl_win_totals_2025_week_11 (1).csv | Game outcome/result - prediction target |
| `Total Result` | high | current_season/strength_of_schedule_nfelo_2025_week_11.csv | Game outcome/result - prediction target |
| `result` | high | cache/schedules_2025.csv | Game outcome/result - prediction target |

---

## 3. High Leakage Risk Columns ⚠️

**These columns must be EXCLUDED from pre-game models:**

| Column Name | Role | Notes |
|-------------|------|-------|
| `Avg Margin` | target_outcome | Game outcome/result - prediction target |
| `Result` | target_outcome | Game outcome/result - prediction target |
| `Total Result` | target_outcome | Game outcome/result - prediction target |
| `result` | target_outcome | Game outcome/result - prediction target |

---

## 4. Columns Requiring Manual Review

**118 columns need manual classification:**

| Column Name | Current Role | Leakage Risk | Example Values | Notes |
|-------------|--------------|--------------|----------------|-------|
| `/ Catch` | meta_misc | unknown | 0.359246, 0.693648, 0.853612, 0.901315, 1.17522 | Could not classify - manual review needed |
| `/ DB` | meta_misc | unknown | 0.249, 0.2533, 0.2591, 0.2658, 0.2681 | Could not classify - manual review needed |
| `/ Game` | meta_misc | unknown | 3.03363, 4.93261, 5.97529, 6.39841, 7.43585 | Could not classify - manual review needed |
| `/ Route` | meta_misc | unknown |  | Could not classify - manual review needed |
| `/ Target` | meta_misc | unknown | 0.255165, 0.541384, 0.632677, 0.752998, 0.799802 | Could not classify - manual review needed |
| `1st Downs` | meta_misc | unknown | 37, 40, 41, 42, 45 | Could not classify - manual review needed |
| `3rd Down` | meta_misc | unknown | 0.235294, 0.293333, 0.296875, 0.323944, 0.344262 | Could not classify - manual review needed |
| `ANY/A` | meta_misc | unknown | 7.0934, 7.4539, 8.2027, 8.3515, 8.5373 | Could not classify - manual review needed |
| `ATS %` | meta_misc | unknown | 0.523077, 0.527157, 0.531773, 0.540948, 0.541806 | Could not classify - manual review needed |
| `Actual` | meta_misc | unknown | 4, 5, 6, 7, 8 | Could not classify - manual review needed |
| `Against` | meta_misc | unknown | 17.0, 17.6667, 19.1111, 21.3333, 22.2222 | Could not classify - manual review needed |
| `All` | meta_misc | unknown | 0.217523, 0.292593, 0.311787, 0.335423, 0.388128 | Could not classify - manual review needed |
| `Atts` | meta_misc | unknown | 147, 262, 273, 308, 315 | Could not classify - manual review needed |
| `Avg PA` | meta_misc | unknown | 19.3592, 20.0, 20.0406, 20.583, 22.4488 | Could not classify - manual review needed |
| `Avg PF` | meta_misc | unknown | 22.9094, 23.4945, 24.5761, 24.7637, 26.8284 | Could not classify - manual review needed |
| `CB` | meta_misc | unknown | 0, 1, 2, 3, 4 | Could not classify - manual review needed |
| `CB Opps` | meta_misc | unknown | 0, 1, 2, 3, 4 | Could not classify - manual review needed |
| `CB%` | meta_misc | unknown | 0.0, 0.25, 0.3333, 0.5, 0.6667 | Could not classify - manual review needed |
| `CPOE` | meta_misc | unknown | 2.1546, 4.3838, 4.6111, 6.1128, 8.5224 | Could not classify - manual review needed |
| `CROE` | meta_misc | unknown | 10.0144, 10.5609, 14.0937, 16.1439, 4.43578 | Could not classify - manual review needed |
| `Carries` | meta_misc | unknown | 32, 39, 48, 5, 6 | Could not classify - manual review needed |
| `Catch Rate` | meta_misc | unknown | 0.680556, 0.71028, 0.741176, 0.780488, 0.835443 | Could not classify - manual review needed |
| `Coach` | meta_misc | unknown | Andy Reid, John Harbaugh, Matt LaFleur, Mike Tomli... | Could not classify - manual review needed |
| `Comp` | meta_misc | unknown | 103, 185, 202, 204, 208 | Could not classify - manual review needed |
| `Comp%` | meta_misc | unknown | 0.6476, 0.6753, 0.7007, 0.7061, 0.7399 | Could not classify - manual review needed |
| `Conf Champ` | meta_misc | unknown | 0, 1, 2, 6 | Could not classify - manual review needed |
| `Dif` | meta_misc | unknown | 10.8889, 11.4444, 2.8889, 8.4444, 9.2222 | Could not classify - manual review needed |
| `Dog` | meta_misc | unknown | 0.575758, 0.581818, 0.585106, 0.61194, 0.62037 | Could not classify - manual review needed |
| `Favorite` | meta_misc | unknown | 0.478049, 0.48, 0.507317, 0.513678, 0.525 | Could not classify - manual review needed |
| `Film` | meta_misc | unknown | 4, 5, 6, 7, 9 | Could not classify - manual review needed |
| `For` | meta_misc | unknown | 24.2222, 26.1111, 27.8889, 30.5556, 31.4444 | Could not classify - manual review needed |
| `GP` | meta_misc | unknown | 10, 5, 7, 8, 9 | Could not classify - manual review needed |
| `Games` | meta_misc | unknown | 303, 309, 320, 322, 474 | Could not classify - manual review needed |
| `Hold` | meta_misc | unknown | 0.0414079, 0.0433604, 0.0444444, 0.0463822, 0.0484... | Could not classify - manual review needed |
| `INT%` | meta_misc | unknown | 0.0065, 0.0068, 0.011, 0.0159, 0.0191 | Could not classify - manual review needed |
| `INTs` | meta_misc | unknown | 1, 2, 3, 5, 6 | Could not classify - manual review needed |
| `INTs.1` | meta_misc | unknown | -0.0006, 0.025, 0.038, 0.0487, 0.05 | Could not classify - manual review needed |
| `Inc` | meta_misc | unknown | -0.0115, 0.0065, 0.0218, 0.0267, 0.039 | Could not classify - manual review needed |
| `L` | meta_misc | unknown | 0, 2, 3, 4, 5 | Could not classify - manual review needed |
| `Losses` | meta_misc | unknown | 116, 120, 122, 138, 167 | Could not classify - manual review needed |
| `Matchup` | meta_misc | unknown | Baltimore Ravens at Cleveland Browns, Carolina Pan... | Could not classify - manual review needed |
| `Opponent` | meta_misc | unknown | -0.462154, 1.14725, 1.32143, 1.39165, 1.73482 | Could not classify - manual review needed |
| `Over` | meta_misc | unknown | -125, -150, 105, 110, 125 | Could not classify - manual review needed |
| `Over.1` | meta_misc | unknown | 0.425532, 0.457256, 0.467532, 0.532468, 0.574468 | Could not classify - manual review needed |
| `Pass` | meta_misc | unknown | 0.1164, 0.2212, 0.2673, 0.277, 0.2823 | Could not classify - manual review needed |
| `Pass.1` | meta_misc | unknown | -0.0344, -0.0621, -0.0678, -0.0828, 0.0004 | Could not classify - manual review needed |
| `Passer Rtg` | meta_misc | unknown | 106.075, 115.544, 117.697, 127.055, 98.4987 | Could not classify - manual review needed |
| `Penalties` | meta_misc | unknown | -0.0101, -0.0166, 0.0011, 0.0026, 0.0214 | Could not classify - manual review needed |
| `Play` | meta_misc | unknown | 0.0505, 0.096, 0.1188, 0.146, 0.1693 | Could not classify - manual review needed |
| `Play.1` | meta_misc | unknown | -0.0014, -0.0328, -0.0782, -0.1205, 0.026 | Could not classify - manual review needed |
| `Play.2` | meta_misc | unknown | 0.0519, 0.1432, 0.1788, 0.1969, 0.2165 | Could not classify - manual review needed |
| `Player` | meta_misc | unknown | 00-0036900, 00-0036963, 00-0037247, 00-0038543, 00... | Could not classify - manual review needed |
| `Points` | meta_misc | unknown | 1.77355, 2.01285, 2.72162, 2.96073, 3.23086 | Could not classify - manual review needed |
| `Pythag` | meta_misc | unknown | 5.1722, 6.2533, 6.4462, 6.7728, 6.8735 | Could not classify - manual review needed |
| `QB` | meta_misc | unknown | Jalen Hurts, Jared Goff, Jordan Love, Josh Allen, ... | Could not classify - manual review needed |
| `QB Adj` | meta_misc | unknown | 1.8485, 11.7198, 12.3302, 27.4697, 9.1518 | Could not classify - manual review needed |
| `Recs` | meta_misc | unknown | 49, 63, 64, 66, 76 | Could not classify - manual review needed |
| `Red Zone` | meta_misc | unknown | 0.153846, 0.156863, 0.211538, 0.318182, 0.404762 | Could not classify - manual review needed |
| `Return (%)` | meta_misc | unknown | -0.00134183, 0.00625, 0.0147102, 0.0320292, 0.0339... | Could not classify - manual review needed |
| `Return (units)` | meta_misc | unknown | -0.4, 11.3, 16.7, 2.2, 5.0 | Could not classify - manual review needed |
| `Rush` | meta_misc | unknown | -0.0039526, -0.0134967, -0.0903789, -0.121129, 0.0... | Could not classify - manual review needed |
| `Rush.1` | meta_misc | unknown | -0.00806835, -0.0748147, -0.107043, -0.157773, 0.0... | Could not classify - manual review needed |
| `Rushing` | meta_misc | unknown | -0.0265, -0.0472, 0.0164, 0.0436, 0.0514 | Could not classify - manual review needed |
| `Starts` | meta_misc | unknown | 10, 2, 5, 6, 9 | Could not classify - manual review needed |
| `Status` | meta_misc | unknown | True | Could not classify - manual review needed |
| `Success` | meta_misc | unknown | 0.5243, 0.5301, 0.5345, 0.539, 0.5429 | Could not classify - manual review needed |
| `Super Bowls` | meta_misc | unknown | 0, 1, 3 | Could not classify - manual review needed |
| `T` | meta_misc | unknown | 0, 1 | Could not classify - manual review needed |
| `TD%` | meta_misc | unknown | 0.054, 0.0573, 0.0733, 0.0812, 0.102 | Could not classify - manual review needed |
| `TD%-INT%` | meta_misc | unknown | 0.0381, 0.0382, 0.0623, 0.0747, 0.0952 | Could not classify - manual review needed |
| `TDs` | meta_misc | unknown | 15, 16, 17, 2, 20 | Could not classify - manual review needed |
| `TDs.1` | meta_misc | unknown | 0, 1, 4, 5, 7 | Could not classify - manual review needed |
| `Target Rate` | meta_misc | unknown |  | Could not classify - manual review needed |
| `Targets` | meta_misc | unknown | 107, 72, 79, 82, 85 | Could not classify - manual review needed |
| `Ties` | meta_misc | unknown | 0, 1, 2 | Could not classify - manual review needed |
| `Time (ET)` | meta_misc | unknown | 1:00PM, 4:05PM, 4:25PM, 8:15PM, 9:30AM | Could not classify - manual review needed |
| `Under` | meta_misc | unknown | -125, -130, -150, 105, 125 | Could not classify - manual review needed |
| `Under.1` | meta_misc | unknown | 0.425532, 0.467532, 0.532468, 0.542744, 0.574468 | Could not classify - manual review needed |
| `W` | meta_misc | unknown | 2, 3, 5, 6, 7 | Could not classify - manual review needed |
| `WPA / DB` | meta_misc | unknown | 0.0048, 0.0053, 0.006, 0.0085, 0.0096 | Could not classify - manual review needed |
| `Win %` | meta_misc | unknown | 0.568323, 0.61165, 0.6125, 0.617162, 0.64557 | Could not classify - manual review needed |
| `Win Prob.` | meta_misc | unknown | 50.6694069, 54.16367499, 62.06944233, 64.12499984,... | Could not classify - manual review needed |
| `Wins` | meta_misc | unknown | 183, 187, 189, 196, 306 | Could not classify - manual review needed |
| `WoW` | meta_misc | unknown | 0.525772, 0.696909, 0.700825, 1.67778, 2.46035 | Could not classify - manual review needed |
| `X.1` | meta_misc | unknown | Chiefs, Lions, Rams, Seahawks, Team | Could not classify - manual review needed |
| `X.2` | meta_misc | unknown | AFCS, AFCW, Div., Jared Goff, Josh Allen | Could not classify - manual review needed |
| `X.20` | meta_misc | unknown | 320.330558, 353.880884, 366.381982, 396.270789, Pl... | Could not classify - manual review needed |
| `X.23` | meta_misc | unknown | 0.038498874, 0.049051094, 0.053486068, 3.55E-02, V... | Could not classify - manual review needed |
| `X.25` | meta_misc | unknown | 26.56866663, 26.64333247, 29.67928801, 35.75231654... | Could not classify - manual review needed |
| `X.26` | meta_misc | unknown | 52.98107611, 53.29503677, 55.39457077, 61.49465415... | Could not classify - manual review needed |
| `X.3` | meta_misc | unknown | 0.558504783, 2.260843771, 4.804841963, 4.835937439... | Could not classify - manual review needed |
| `X.4` | meta_misc | unknown | 29, 30, 31, 37, 6.326224547 | Could not classify - manual review needed |
| `X.5` | meta_misc | unknown | 70.2, 88.3, 95.1, 96.3, PO% | Could not classify - manual review needed |
| `X.6` | meta_misc | unknown | 25.8, 35.9, 56.7, 64.1, Div% | Could not classify - manual review needed |
| `X.7` | meta_misc | unknown | 14.3, 16.4, 18.1, 25.9, Cnf% | Could not classify - manual review needed |
| `X.8` | meta_misc | unknown | 10.1, 16.8, 7.9, 8.8, SB% | Could not classify - manual review needed |
| `X.9` | meta_misc | unknown | 10, 6, 7, 9, Prim. | Could not classify - manual review needed |
| `YAC` | meta_misc | unknown | 0.0023, 0.0037, 0.0716, 0.0788, 0.1885 | Could not classify - manual review needed |
| `YPA` | meta_misc | unknown | 7.4571, 7.8799, 8.1641, 8.1868, 8.4966 | Could not classify - manual review needed |
| `YPC` | meta_misc | unknown | 10.8281, 10.9342, 11.7424, 15.5918, 16.5238 | Could not classify - manual review needed |
| `YPG` | meta_misc | unknown | 115.667, 77.0, 84.8889, 92.3333, 96.875 | Could not classify - manual review needed |
| `YTD` | meta_misc | unknown | 0.204673, 2.64027, 2.85775, 5.08876, 6.2258 | Could not classify - manual review needed |
| `aDOT` | meta_misc | unknown | 11.8333, 12.8353, 6.381, 7.65854, 7.7939 | Could not classify - manual review needed |
| `div_game` | meta_misc | unknown | 0, 1 | Could not classify - manual review needed |
| `espn` | meta_misc | unknown | 401772510, 401772714, 401772719, 401772829, 401772... | Could not classify - manual review needed |
| `ftn` | meta_misc | unknown | 6734.0, 6735.0, 6736.0, 6737.0, 6738.0 | Could not classify - manual review needed |
| `game_type` | meta_misc | unknown | REG | Could not classify - manual review needed |
| `gameday` | meta_misc | unknown | 2025-09-04, 2025-09-05, 2025-09-07, 2025-09-08, 20... | Could not classify - manual review needed |
| `gametime` | meta_misc | unknown | 13:00, 16:05, 16:25, 20:00, 20:20 | Could not classify - manual review needed |
| `gsis` | meta_misc | unknown | 59843.0, 59844.0, 59845.0, 59846.0, 59847.0 | Could not classify - manual review needed |
| `overtime` | meta_misc | unknown | 0.0, 1.0 | Could not classify - manual review needed |
| `pff` | meta_misc | unknown | 28418.0, 28419.0, 28420.0, 28421.0, 28422.0 | Could not classify - manual review needed |
| `pfr` | meta_misc | unknown | 202509040phi, 202509050sdg, 202509070atl, 20250907... | Could not classify - manual review needed |
| `referee` | meta_misc | unknown | Adrian Hill, Brad Allen, Carl Cheffers, Land Clark... | Could not classify - manual review needed |
| `temp` | meta_misc | unknown | 62.0, 63.0, 67.0, 75.0, 85.0 | Could not classify - manual review needed |
| `vs Sticks` | meta_misc | unknown | -0.0413, -0.6718, -1.0476, -1.8718, 0.974 | Could not classify - manual review needed |
| `~~~2025 Forecast~~~` | meta_misc | unknown | 10.408, 11.239, 12.092, 12.525, Avg. Wins | Could not classify - manual review needed |
| `~~~Per 17 Gms~~~` | meta_misc | unknown | 103.6807881, 104.6155951, 92.01385717, 97.93925609... | Could not classify - manual review needed |

---

## 5. Recommendations for Ball_Knower v2.0

### ✅ SAFE for Pre-Game Models

Use these column types freely in v2.0:

- **id_key** (19 columns)
- **pre_game_market** (17 columns)
- **pre_game_team_strength** (14 columns)
- **pre_game_structure** (20 columns)

### ⚠️ VERIFY BEFORE USE

These 15 columns need timing verification:

- **in_game_stats** - Ensure these are historical aggregates, not game-specific actuals

### 🚫 EXCLUDE from Pre-Game Models

These 4 columns contain post-game information:

- **target_outcome** (4 columns)

### 📋 Manual Review Required

Review 118 unclassified columns (see Section 4)

---

*Generated by `tools/build_feature_catalog.py`*
