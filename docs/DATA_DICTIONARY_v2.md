# Ball Knower v2.0 - Data Dictionary

*Auto-generated from schema analysis, feature catalog, and file mapping tools*

## 1. Overview

- **Total Columns**: 30
- **Source Files**: 1
- **Roles Distribution**:
  - `id_key`: 3 columns
  - `outcome`: 6 columns
  - `pre_game_context`: 7 columns
  - `pre_game_market`: 4 columns
  - `pre_game_team_strength`: 6 columns
  - `structure`: 4 columns
- **Leakage Risk Distribution**:
  - `low`: 22 columns
  - `medium`: 2 columns
  - `high`: 6 columns

## 2. Core ID/Key Columns

| Column Name | Source Files | Dtype | Notes |
|-------------|--------------|-------|-------|
| `away_team` | team_week_epa_2013_2024.csv... | string | Team identifier - safe for modeling |
| `game_id` | team_week_epa_2013_2024.csv... | string | Unique identifier safe for all uses |
| `home_team` | team_week_epa_2013_2024.csv... | string | Team identifier - safe for modeling |

## 3. Safe Pre-Game Features

*Columns with `leakage_risk = low`, safe for pre-game modeling*

### Id Key

| Column Name | Dtype | Missing % | Notes |
|-------------|-------|-----------|-------|
| `away_team` | string | 0.0 | Team identifier - safe for modeling |
| `game_id` | string | 0.0 | Unique identifier safe for all uses |
| `home_team` | string | 0.0 | Team identifier - safe for modeling |

### Pre Game Context

| Column Name | Dtype | Missing % | Notes |
|-------------|-------|-----------|-------|
| `away_rest_days` | int | 10.0% | Rest days known before game |
| `home_rest_days` | int | 10.0% | Rest days known before game |
| `roof` | category | 25.0% | Roof type known before game |
| `stadium` | string | 25.0% | Stadium known before game |
| `surface` | category | 30.0% | Surface type known before game |

### Pre Game Market

| Column Name | Dtype | Missing % | Notes |
|-------------|-------|-----------|-------|
| `away_moneyline` | float | 18.0% | Moneyline odds available before kickoff |
| `home_moneyline` | float | 18.0% | Moneyline odds available before kickoff |
| `spread_line` | float | 15.0% | Vegas spread available before kickoff |
| `total_line` | float | 15.0% | Vegas total available before kickoff |

### Pre Game Team Strength

| Column Name | Dtype | Missing % | Notes |
|-------------|-------|-----------|-------|
| `away_elo_pre` | float | 8.0% | Elo rating before game - safe for pre-game models |
| `away_epa_defense` | float | 5.0% | Defensive EPA entering the game - safe for pre-game models |
| `away_epa_offense` | float | 5.0% | Offensive EPA entering the game - safe for pre-game models |
| `home_elo_pre` | float | 8.0% | Elo rating before game - safe for pre-game models |
| `home_epa_defense` | float | 5.0% | Defensive EPA entering the game - safe for pre-game models |
| `home_epa_offense` | float | 5.0% | Offensive EPA entering the game - safe for pre-game models |

### Structure

| Column Name | Dtype | Missing % | Notes |
|-------------|-------|-----------|-------|
| `div_game` | bool | 0.0 | Divisional matchup flag - known before game |
| `gameday` | datetime | 0.0 | Game date - structural variable |
| `season` | int | 0.0 | Season year - structural variable |
| `week` | int | 0.0 | Week number - structural variable |

## 4. High / Medium Leakage Features

*Columns that may contain in-game or post-game information*

| Column Name | Leakage Risk | Role | Reason |
|-------------|--------------|------|--------|
| `away_score` | **high** | outcome | LEAKAGE: Final score only known after game |
| `home_score` | **high** | outcome | LEAKAGE: Final score only known after game |
| `home_win` | **high** | outcome | LEAKAGE: Binary outcome only known after game |
| `margin` | **high** | outcome | LEAKAGE: Derived from final scores |
| `result` | **high** | outcome | LEAKAGE: Game result only known after game |
| `total_points` | **high** | outcome | LEAKAGE: Derived from final scores |
| `temp` | **medium** | pre_game_context | Weather forecast available but may change - verify timing |
| `wind` | **medium** | pre_game_context | Wind forecast available but may change - verify timing |

## 5. Ambiguous / Unknown Columns

*Columns requiring manual review*

*No ambiguous columns - all columns have been classified*

## 6. Recommendations

### Safe for v2.0 Pre-Game Models

- **22 columns** identified as safe (low leakage risk)
- Focus on columns with roles: `structure`, `pre_game_market`, `pre_game_team_strength`
- Key safe features:
  - `away_elo_pre`
  - `away_epa_defense`
  - `away_epa_offense`
  - `away_moneyline`
  - `away_rest_days`
  - `away_team`
  - `div_game`
  - `game_id`
  - `gameday`
  - `home_elo_pre`

### Must Exclude from Pre-Game Models

- **6 columns** with high leakage risk must be excluded
- High-risk columns to avoid:
  - `away_score` - LEAKAGE: Final score only known after game
  - `home_score` - LEAKAGE: Final score only known after game
  - `home_win` - LEAKAGE: Binary outcome only known after game
  - `margin` - LEAKAGE: Derived from final scores
  - `result` - LEAKAGE: Game result only known after game
  - `total_points` - LEAKAGE: Derived from final scores

### Require Manual Review

- **2 columns** with medium leakage risk need timing verification
- **0 columns** with unknown risk need classification
- Medium-risk columns to verify:
  - `temp` - verify timing: Weather forecast available but may change - verify
  - `wind` - verify timing: Wind forecast available but may change - verify ti

