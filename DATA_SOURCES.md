# Ball Knower Data Sources

## Overview

This document provides detailed descriptions of all data sources used in the Ball Knower system. Each source is organized by **category** (what kind of data) and **provider** (where it comes from), following the category-first naming convention.

## Quick Reference

| Category | Providers | Update Frequency | Use Case |
|----------|-----------|------------------|----------|
| power_ratings | nfelo, 538, espn, pff | Weekly | Overall team strength |
| epa_tiers | nfelo, gsis | Weekly | Play efficiency metrics |
| strength_of_schedule | nfelo, 538 | Weekly | Opponent difficulty |
| qb_epa | nfelo, pff, gsis | Weekly | QB performance |
| qb_rankings | nfelo, pff | Weekly | QB quality ratings |
| weekly_projections_ppg | nfelo, 538 | Weekly | Game forecasts |
| weekly_projections_elo | nfelo, 538 | Weekly | ELO-based projections |
| nfl_receiving_leaders | nfelo, gsis | Weekly | Season stats |
| nfl_win_totals | nfelo, 538 | Weekly | Season projections |

---

## Category: power_ratings

### power_ratings_nfelo

**File:** `power_ratings_nfelo_{season}_week_{week}.csv`

**Provider:** nfeloapp.com

**Description:** Comprehensive power ratings combining ELO methodology with QB adjustments and recent performance.

**Columns:**
- `team` - Team abbreviation (normalized)
- `rating` - Overall nfelo rating (scale: ~1400-1700)
- `elo` - Base ELO rating
- `qb_adj` - QB adjustment factor
- `off_rating` - Offensive rating component
- `def_rating` - Defensive rating component

**Update Schedule:** Tuesday after MNF (weekly)

**Use Cases:**
- Primary team strength metric
- Base for spread predictions
- Historical performance tracking

**Quality Notes:**
- High consistency with Vegas lines
- Well-calibrated for NFL
- Accounts for QB changes
- Regresses to mean across seasons

---

### power_ratings_substack

**File:** `power_ratings_substack_{season}_week_{week}.csv`

**Provider:** Multiple Substack NFL modelers (aggregated)

**Description:** Independent power ratings from Substack analysts, providing alternative perspectives on team strength.

**Columns:**
- `team` - Team abbreviation (normalized)
- `off_rating` - Offensive rating
- `def_rating` - Defensive rating
- `overall_rating` - Combined overall rating

**Update Schedule:** Varies by analyst (weekly)

**Use Cases:**
- Ensemble modeling (diversify signal)
- Validation of nfelo ratings
- Alternative team strength metric

**Quality Notes:**
- Less standardized than nfelo
- May require additional normalization
- Useful for consensus/disagreement analysis
- Header row artifacts (handled by loader)

**Migration Note:** Future versions will decompose `power_ratings_substack` into specific providers (e.g., `power_ratings_user`, `power_ratings_manual`) based on source tracking.

---

### power_ratings_538

**File:** `power_ratings_538_{season}_week_{week}.csv`

**Provider:** FiveThirtyEight (when available)

**Description:** FiveThirtyEight's ELO-based power ratings with QB adjustments and market-based calibration.

**Columns:**
- `team` - Team abbreviation
- `elo` - Current ELO rating
- `qb_adj` - QB value adjustment
- `rating` - Overall power rating

**Update Schedule:** Weekly (when 538 publishes forecasts)

**Use Cases:**
- Cross-validation with nfelo
- Ensemble modeling
- Historical backtesting

**Quality Notes:**
- Well-documented methodology
- Historically strong performance
- May not be available for all weeks

---

## Category: epa_tiers

### epa_tiers_nfelo

**File:** `epa_tiers_nfelo_{season}_week_{week}.csv`

**Provider:** nfeloapp.com

**Description:** Expected Points Added (EPA) metrics showing team efficiency per play on offense and defense.

**Columns:**
- `team` - Team abbreviation
- `off_epa` - Offensive EPA per play
- `def_epa` - Defensive EPA per play (allowed)
- `off_pass_epa` - Passing EPA per play
- `off_run_epa` - Rushing EPA per play
- `def_pass_epa` - Defensive passing EPA allowed
- `def_run_epa` - Defensive rushing EPA allowed
- `success_rate` - Percentage of successful plays

**Update Schedule:** Tuesday after MNF

**Use Cases:**
- Feature for spread prediction
- Team efficiency analysis
- Matchup analysis (off vs def)

**Quality Notes:**
- Highly predictive of point differentials
- Normalizes for game situation
- Accounts for opponent strength
- EPA scale: ~-0.2 to +0.2 per play

**Key Insight:** EPA differential × 100 ≈ expected point differential per game

---

### epa_tiers_gsis

**File:** `epa_tiers_gsis_{season}_week_{week}.csv`

**Provider:** NFL Game Statistics & Information System (official stats)

**Description:** Official NFL EPA calculations from play-by-play data.

**Columns:**
- `team` - Team abbreviation
- `off_epa` - Offensive EPA per play
- `def_epa` - Defensive EPA per play
- Additional granular metrics

**Update Schedule:** Weekly (official NFL data release)

**Use Cases:**
- Ground truth EPA validation
- Historical backtesting
- Official metric tracking

**Quality Notes:**
- Authoritative source
- May differ slightly from nfelo due to methodology
- Available through nfl_data_py package

---

## Category: strength_of_schedule

### strength_of_schedule_nfelo

**File:** `strength_of_schedule_nfelo_{season}_week_{week}.csv`

**Provider:** nfeloapp.com

**Description:** Opponent difficulty metrics based on strength of past and future opponents.

**Columns:**
- `team` - Team abbreviation
- `sos` - Strength of schedule (past opponents)
- `sos_remaining` - Remaining schedule strength
- `sos_rank` - SOS rank (1 = hardest)

**Update Schedule:** Tuesday after MNF

**Use Cases:**
- Context for team record
- Season projections
- Win total betting
- Identifying undervalued teams

**Quality Notes:**
- Based on opponent ELO ratings
- Updates as season progresses
- Useful for regression analysis

---

## Category: qb_epa

### qb_epa_nfelo

**File:** `qb_epa_nfelo_{season}_week_{week}.csv`

**Provider:** nfeloapp.com

**Description:** Quarterback-specific EPA metrics and performance grades.

**Columns:**
- `player` - QB name
- `team` - Team abbreviation
- `qb_epa` - QB EPA per play
- `completions` - Pass completions
- `attempts` - Pass attempts
- `yards` - Passing yards
- `td` - Touchdown passes
- `int` - Interceptions
- `success_rate` - Play success rate

**Update Schedule:** Tuesday after MNF

**Use Cases:**
- QB-adjusted spread predictions
- Injury impact analysis
- Backup QB downgrades

**Quality Notes:**
- Isolates QB contribution to EPA
- Accounts for supporting cast
- Useful for QB change scenarios

---

### qb_epa_substack

**File:** `qb_epa_substack_{season}_week_{week}.csv`

**Provider:** Substack NFL analysts

**Description:** Alternative QB EPA metrics from independent modelers.

**Columns:**
- `player` - QB name
- `team` - Team abbreviation (note: may use 'Tms' column)
- `qb_epa` - QB EPA metric
- Additional passing stats

**Update Schedule:** Weekly (varies by analyst)

**Use Cases:**
- Ensemble QB metrics
- Validation of nfelo QB EPA
- Additional QB context

**Quality Notes:**
- Header artifacts (handled by loader)
- May use lowercase team codes (normalized)
- Multi-team QBs: takes first team

---

### qb_epa_pff

**File:** `qb_epa_pff_{season}_week_{week}.csv`

**Provider:** Pro Football Focus (when available)

**Description:** PFF's QB grades and EPA metrics incorporating advanced charting data.

**Columns:**
- `player` - QB name
- `team` - Team abbreviation
- `grade` - PFF overall QB grade (0-100)
- `epa` - EPA per play
- Additional PFF-specific metrics

**Update Schedule:** Weekly (subscription required)

**Use Cases:**
- High-quality QB performance metric
- Injury/backup impact estimation
- Cross-validation with other QB metrics

---

## Category: qb_rankings

### qb_rankings_nfelo

**File:** `qb_rankings_nfelo_{season}_week_{week}.csv`

**Provider:** nfeloapp.com

**Description:** Comprehensive QB rankings with stats and grades.

**Columns:**
- `rank` - Overall QB rank
- `player` - QB name
- `team` - Team abbreviation
- Various performance metrics

**Update Schedule:** Weekly

**Use Cases:**
- Reference table for QB quality
- Model feature (QB rank)
- Narrative/reporting

---

## Category: weekly_projections_ppg

### weekly_projections_ppg_nfelo

**File:** `weekly_projections_ppg_nfelo_{season}_week_{week}.csv`

**Provider:** nfeloapp.com

**Description:** Points per game projections for upcoming matchups.

**Columns:**
- `home_team` - Home team abbreviation
- `away_team` - Away team abbreviation
- `home_ppg` - Projected home points
- `away_ppg` - Projected away points
- `spread` - Projected spread (negative = home favored)
- `total` - Projected total points

**Update Schedule:** Weekly

**Use Cases:**
- Baseline spread predictions
- Total (over/under) betting
- Comparison with Ball Knower model

---

### weekly_projections_ppg_substack

**File:** `weekly_projections_ppg_substack_{season}_week_{week}.csv`

**Provider:** Substack NFL modelers

**Description:** Alternative game projections from independent analysts.

**Columns:**
- Matchup identifiers (home/away teams)
- Point projections
- Spread projections

**Update Schedule:** Weekly

**Use Cases:**
- Ensemble game forecasts
- Model validation
- Consensus analysis

**Quality Notes:**
- Format may vary by analyst
- Header artifacts (handled by loader)
- May not have standardized `team` column

---

## Category: weekly_projections_elo

### weekly_projections_elo_nfelo

**File:** `weekly_projections_elo_nfelo_{season}_week_{week}.csv`

**Provider:** nfeloapp.com

**Description:** ELO-based win probability projections for matchups.

**Columns:**
- `home_team` - Home team
- `away_team` - Away team
- `home_elo` - Home team ELO
- `away_elo` - Away team ELO
- `win_prob` - Home win probability

**Update Schedule:** Weekly

**Use Cases:**
- Moneyline betting
- Win probability analysis
- ELO tracking

**Note:** Currently may be embedded in `substack_weekly_proj_elo` files (legacy naming).

---

## Category: nfl_receiving_leaders

### nfl_receiving_leaders_nfelo

**File:** `nfl_receiving_leaders_nfelo_{season}_week_{week}.csv`

**Provider:** nfeloapp.com

**Description:** Season-to-date receiving statistics leaders.

**Columns:**
- `player` - Receiver name
- `team` - Team abbreviation
- `receptions` - Total receptions
- `yards` - Receiving yards
- `td` - Receiving touchdowns
- Additional stats

**Update Schedule:** Weekly

**Use Cases:**
- Reference table
- Feature engineering (target quality)
- Player prop betting context

**Note:** Not typically used in spread models directly, but useful for context.

---

## Category: nfl_win_totals

### nfl_win_totals_nfelo

**File:** `nfl_win_totals_nfelo_{season}_week_{week}.csv`

**Provider:** nfeloapp.com

**Description:** Season win total projections based on current standings and remaining schedule.

**Columns:**
- `team` - Team abbreviation
- `wins` - Current wins
- `losses` - Current losses
- `projected_wins` - Season win total projection
- `playoff_prob` - Playoff probability

**Update Schedule:** Weekly

**Use Cases:**
- Season win total betting
- Playoff race analysis
- Futures betting context

---

## Reference Data

### nfl_head_coaches.csv

**File:** `data/reference/nfl_head_coaches.csv`

**Provider:** Manual curation

**Description:** Head coach information and tenure.

**Columns:**
- `team` - Team abbreviation
- `coach` - Coach name
- `hire_year` - Year hired
- `win_rate` - Career win rate
- `playoff_wins` - Playoff wins

**Update Schedule:** Seasonal (as coaching changes occur)

**Use Cases:**
- Coaching change features
- Contextual analysis

---

## Historical Play-by-Play Data

### nfl_data_py Package

**Provider:** nflverse (nfl_data_py Python package)

**Description:** Comprehensive historical play-by-play data from 1999-present.

**Access:**
```python
import nfl_data_py as nfl
pbp = nfl.import_pbp_data(range(2015, 2025))
schedules = nfl.import_schedules(range(2015, 2025))
```

**Data Includes:**
- Every play from 1999-present
- Game schedules and results
- Vegas lines (spread, total, moneyline)
- Weather data
- Stadium information
- Referee assignments

**Use Cases:**
- Historical backtesting
- Feature engineering
- Model training
- Performance validation

**Quality Notes:**
- Authoritative source for NFL play-by-play
- Regularly updated
- Well-documented schemas
- Used by NFL analytics community

**Setup Note:** Due to network restrictions in some environments, pre-aggregate to team-week statistics. See `DATA_SETUP_GUIDE.md` for details.

---

## Adding New Data Sources

### Process

1. **Determine category:** Does your data fit an existing category (power_ratings, epa_tiers, etc.)? If not, define a new category in `BALL_KNOWER_SPEC.md`.

2. **Name the file:** Use category-first naming:
   ```
   {category}_{provider}_{season}_week_{week}.csv
   ```

3. **Normalize team names:** Ensure the `team` column uses canonical abbreviations (see `BALL_KNOWER_SPEC.md`).

4. **Place in directory:** Put file in `data/current_season/` (or custom directory).

5. **Test loading:**
   ```python
   from ball_knower.io import loaders
   df = loaders.load_power_ratings("your_provider", 2025, 11)
   ```

No code changes needed - the loader auto-discovers files following the naming convention!

### Example: Adding ESPN FPI

1. Download ESPN FPI data
2. Create file: `power_ratings_espn_2025_week_11.csv`
3. Ensure columns include: `team`, `rating` (or `fpi`)
4. Normalize team names to NFL abbreviations
5. Place in `data/current_season/`
6. Load with:
   ```python
   fpi = loaders.load_power_ratings("espn", 2025, 11)
   ```

---

## Data Quality Checklist

Before adding a new data source, verify:

- [ ] File follows category-first naming convention
- [ ] CSV format with comma delimiters
- [ ] Header row present (or in second row if multi-row header)
- [ ] Team column uses canonical abbreviations or mappable names
- [ ] No special characters in column names
- [ ] Numeric columns properly formatted (not strings)
- [ ] No duplicate team entries (unless player-level data)
- [ ] File placed in correct directory
- [ ] Loads successfully with appropriate loader function

---

## Provider Comparison

### Power Ratings Providers

| Provider | Methodology | Strength | Weakness |
|----------|-------------|----------|----------|
| nfelo | ELO + QB adj | Consistent, well-calibrated | Slow to react to changes |
| 538 | ELO + market | Market-aware, robust | Not always available |
| Substack | Varies | Independent perspectives | Variable quality |
| PFF | Grades + analytics | Deep film analysis | Subscription required |

### EPA Providers

| Provider | Methodology | Strength | Weakness |
|----------|-------------|----------|----------|
| nfelo | Play-by-play derived | Fast updates | May differ from official |
| GSIS | Official NFL stats | Authoritative | Less granular |

---

## Historical Provider Migration

### From "substack" to Specific Providers

**Old approach:** All non-nfelo sources labeled as "substack"

**New approach:** Track specific providers (nfelo, 538, espn, pff, gsis, user, manual)

**Migration path:**
1. Identify actual source of each "substack" file
2. Rename to appropriate provider (e.g., `power_ratings_user`)
3. Update documentation to reflect true source
4. Maintain backward compatibility through loader fallback

**Timeline:** Gradual migration as sources are identified and validated.

---

## See Also

- `BALL_KNOWER_SPEC.md` - System specification and naming conventions
- `FEATURE_TIERS.md` - Feature organization for modeling
- `DATA_SETUP_GUIDE.md` - How to set up historical data
- `ball_knower/io/loaders.py` - Loader implementation
