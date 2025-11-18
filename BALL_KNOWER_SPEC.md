# Ball Knower System Specification

## Overview

Ball Knower is a modular NFL spread prediction system that combines multiple data sources through a unified category-first naming convention. This document defines the canonical data schema, naming conventions, and system architecture.

## File Naming Convention

### Primary Format (Category-First)

All weekly data files follow this structure:

```
{category}_{provider}_{season}_week_{week}.csv
```

**Examples:**
- `power_ratings_nfelo_2025_week_11.csv`
- `qb_epa_gsis_2025_week_12.csv`
- `team_efficiency_538_2024_week_18.csv`

### Legacy Format (Provider-First)

For backward compatibility, the system supports legacy naming:

```
{provider}_{category}_{season}_week_{week}.csv
```

**Note:** All new files should use category-first naming. The loader will automatically fall back to provider-first if category-first is not found.

## Canonical Providers

The system recognizes these official providers (replacing "substack" as a top-level identifier):

| Provider | Description | Use Cases |
|----------|-------------|-----------|
| `nfelo` | nfeloapp.com ratings and metrics | Power ratings, EPA tiers, SOS |
| `538` | FiveThirtyEight models | Team efficiency, QB adjustments |
| `espn` | ESPN Analytics | Power rankings, FPI |
| `pff` | Pro Football Focus | Player grades, advanced metrics |
| `gsis` | NFL Game Statistics & Info System | Official NFL stats |
| `user` | Custom user uploads | Manual data entry |
| `manual` | Hand-curated datasets | Reference tables |

**Migration Note:** Existing files with `substack` in the name are mapped to `nfelo` or other appropriate providers based on content.

## Canonical Data Categories

### Team-Level Metrics

#### power_ratings
Overall team strength ratings from various models.

**Required columns:**
- `team` (str): Normalized team abbreviation (e.g., "KC", "BUF")
- `rating` (float): Overall power rating
- `off_rating` (float, optional): Offensive rating
- `def_rating` (float, optional): Defensive rating
- `elo` (float, optional): ELO rating if applicable

**File naming examples:**
- `power_ratings_nfelo_2025_week_11.csv`
- `power_ratings_538_2025_week_11.csv`
- `power_ratings_espn_2025_week_11.csv`

#### epa_tiers
EPA (Expected Points Added) per play metrics.

**Required columns:**
- `team` (str): Normalized team abbreviation
- `off_epa` (float): Offensive EPA per play
- `def_epa` (float): Defensive EPA per play

**Optional columns:**
- `off_pass_epa` (float): Passing EPA per play
- `off_run_epa` (float): Rushing EPA per play
- `def_pass_epa` (float): Defensive passing EPA allowed
- `def_run_epa` (float): Defensive rushing EPA allowed
- `success_rate` (float): Play success rate

**File naming examples:**
- `epa_tiers_nfelo_2025_week_11.csv`
- `epa_tiers_gsis_2025_week_11.csv`

#### strength_of_schedule
Opponent difficulty and schedule strength metrics.

**Required columns:**
- `team` (str): Normalized team abbreviation
- `sos` (float): Strength of schedule metric
- `sos_remaining` (float, optional): Remaining schedule strength

**File naming examples:**
- `strength_of_schedule_nfelo_2025_week_11.csv`
- `strength_of_schedule_538_2025_week_11.csv`

### Player-Level Metrics

#### qb_epa
Quarterback-specific EPA and performance metrics.

**Required columns:**
- `team` (str): Normalized team abbreviation
- `player` (str, optional): Player name
- `qb_epa` (float): QB EPA per play

**Optional columns:**
- `completions` (int): Pass completions
- `attempts` (int): Pass attempts
- `yards` (int): Passing yards
- `td` (int): Touchdowns
- `int` (int): Interceptions
- `success_rate` (float): Play success rate

**File naming examples:**
- `qb_epa_nfelo_2025_week_11.csv`
- `qb_epa_pff_2025_week_11.csv`
- `qb_epa_gsis_2025_week_11.csv`

#### qb_rankings
Comprehensive QB rankings and stats.

**Required columns:**
- `team` (str): Normalized team abbreviation
- `player` (str): Player name
- `rank` (int): Overall ranking

**File naming examples:**
- `qb_rankings_nfelo_2025_week_11.csv`
- `qb_rankings_pff_2025_week_11.csv`

### Game Projections

#### weekly_projections_ppg
Points per game projections and game-level forecasts.

**Required columns:**
- `home_team` (str): Home team abbreviation
- `away_team` (str): Away team abbreviation
- `home_ppg` (float): Projected home points
- `away_ppg` (float): Projected away points

**Optional columns:**
- `spread` (float): Projected spread (negative = home favored)
- `total` (float): Projected total points
- `home_win_prob` (float): Home win probability

**File naming examples:**
- `weekly_projections_ppg_nfelo_2025_week_11.csv`
- `weekly_projections_ppg_538_2025_week_11.csv`

#### weekly_projections_elo
ELO-based game projections.

**Required columns:**
- `home_team` (str): Home team abbreviation
- `away_team` (str): Away team abbreviation
- `home_elo` (float): Home team ELO
- `away_elo` (float): Away team ELO
- `win_prob` (float): Home win probability

**File naming examples:**
- `weekly_projections_elo_nfelo_2025_week_11.csv`
- `weekly_projections_elo_538_2025_week_11.csv`

### Reference Data

#### nfl_receiving_leaders
Season receiving statistics leaders.

**File naming examples:**
- `nfl_receiving_leaders_nfelo_2025_week_11.csv`
- `nfl_receiving_leaders_gsis_2025_week_11.csv`

#### nfl_win_totals
Season win total projections.

**File naming examples:**
- `nfl_win_totals_nfelo_2025_week_11.csv`
- `nfl_win_totals_538_2025_week_11.csv`

## Team Name Normalization

All team columns must use canonical NFL abbreviations from `nfl_data_py`:

| Team | Abbreviation |
|------|--------------|
| Arizona Cardinals | ARI |
| Atlanta Falcons | ATL |
| Baltimore Ravens | BAL |
| Buffalo Bills | BUF |
| Carolina Panthers | CAR |
| Chicago Bears | CHI |
| Cincinnati Bengals | CIN |
| Cleveland Browns | CLE |
| Dallas Cowboys | DAL |
| Denver Broncos | DEN |
| Detroit Lions | DET |
| Green Bay Packers | GB |
| Houston Texans | HOU |
| Indianapolis Colts | IND |
| Jacksonville Jaguars | JAX |
| Kansas City Chiefs | KC |
| Las Vegas Raiders | LV |
| Los Angeles Chargers | LAC |
| Los Angeles Rams | LAR |
| Miami Dolphins | MIA |
| Minnesota Vikings | MIN |
| New England Patriots | NE |
| New Orleans Saints | NO |
| New York Giants | NYG |
| New York Jets | NYJ |
| Philadelphia Eagles | PHI |
| Pittsburgh Steelers | PIT |
| San Francisco 49ers | SF |
| Seattle Seahawks | SEA |
| Tampa Bay Buccaneers | TB |
| Tennessee Titans | TEN |
| Washington Commanders | WAS |

**Historical team mappings:**
- `OAK` → `LV` (Raiders moved in 2020)
- `SD` → `LAC` (Chargers moved in 2017)
- `STL` → `LAR` (Rams moved in 2016)

## Data Loading API

### Primary Interface

```python
from ball_knower.io import loaders

# Load all sources for a given week
data = loaders.load_all_sources(season=2025, week=11)

# Access individual datasets
power_ratings_nfelo = data["power_ratings_nfelo"]
epa_tiers_nfelo = data["epa_tiers_nfelo"]
qb_epa = data["qb_epa_nfelo"]

# Access merged ratings
merged = data["merged_ratings"]
```

### Individual Loaders

```python
# Load specific category + provider
power_ratings = loaders.load_power_ratings("nfelo", 2025, 11)
epa_tiers = loaders.load_epa_tiers("nfelo", 2025, 11)
sos = loaders.load_strength_of_schedule("nfelo", 2025, 11)
qb_epa = loaders.load_qb_epa("nfelo", 2025, 11)
projections = loaders.load_weekly_projections_ppg("nfelo", 2025, 11)
```

### Custom Data Directory

```python
# Override default data directory
data = loaders.load_all_sources(
    season=2025,
    week=11,
    data_dir="/path/to/custom/data"
)
```

## File Resolution Logic

The loader uses this priority order:

1. **Category-first primary:** `{category}_{provider}_{season}_week_{week}.csv`
2. **Provider-first fallback:** `{provider}_{category}_{season}_week_{week}.csv`
3. **Abbreviated categories:** `{provider}_{category_short}_{season}_week_{week}.csv`
   - Example: `weekly_proj_elo` instead of `weekly_projections_elo`

If no file is found, a `FileNotFoundError` is raised with helpful diagnostics.

## CSV Header Handling

The loader automatically handles common CSV quirks:

- **Multi-row headers:** Skips first row if it's all NaN or duplicates column names
- **Excel artifacts:** Removes columns like `X.1`, `X.2`, etc.
- **Case variations:** Renames `Team` → `team` automatically
- **Multi-team players:** For QB data with "cle, cin" format, takes first team

## Data Quality Requirements

### All Files Must:

1. **Have a team column** (or team-identifying columns like `home_team`/`away_team`)
2. **Use canonical team abbreviations** or names that map to them
3. **Be CSV format** with comma delimiters
4. **Have column headers** in first or second row
5. **Follow naming convention** for automatic discovery

### Validation

All loaded data:
- Is normalized to canonical team abbreviations
- Drops rows with unmapped team names (with warning)
- Preserves all original columns
- Returns pandas DataFrames

## Model Integration

### Feature Tier System

Data sources are organized into tiers for model building:

**Tier 1 (Core):** Always available, high quality
- `power_ratings_nfelo`
- `epa_tiers_nfelo`
- `strength_of_schedule_nfelo`

**Tier 2 (Enhanced):** Provider-specific features
- `qb_epa_*`
- `power_ratings_538`
- `team_efficiency_pff`

**Tier 3 (Experimental):** User-contributed or manual
- `power_ratings_user`
- Custom feature engineering

See `FEATURE_TIERS.md` for detailed tier definitions.

### Merge Strategy

The `merge_team_ratings()` function combines all team-level metrics:

1. Start with `power_ratings_nfelo` as base
2. Left-merge each additional source on `team`
3. Apply suffixes to prevent column collisions
4. Return single DataFrame with all features

## Extension Guidelines

### Adding a New Provider

1. Add provider name to canonical list in this document
2. Create files following category-first naming
3. Ensure team column uses standard abbreviations
4. No code changes needed - loader auto-discovers

### Adding a New Category

1. Define category schema in this document
2. Add loader function to `ball_knower/io/loaders.py`:
   ```python
   def load_new_category(provider, season, week, data_dir=None):
       path = _resolve_file("new_category", provider, season, week, data_dir)
       df = pd.read_csv(path)
       # ... normalization logic
       return _normalize_team_column(df, team_col="team")
   ```
3. Add to `load_all_sources()` orchestration
4. Update documentation

## Version History

- **v1.0** (2025-11-18): Initial category-first specification
  - Established canonical naming convention
  - Defined core categories and providers
  - Created unified loader API
  - Deprecated "substack" as top-level identifier

## See Also

- `DATA_SOURCES.md` - Detailed descriptions of each data source
- `FEATURE_TIERS.md` - Feature tier definitions for modeling
- `ball_knower/io/loaders.py` - Loader implementation
- `README.md` - Project overview and quick start
