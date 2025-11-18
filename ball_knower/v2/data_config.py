"""
Ball Knower v2.0 - Data Configuration Module

This module defines the unified data schema, feature tiers, and column mappings
for the v2.0 data pipeline.

Feature Tier System:
- T0: Structural keys (identifiers, no predictive value)
- T1: Safe core features (fundamental team strength metrics)
- T2: Market and situational features (context-dependent metrics)
- T3: Experimental features (complex derived features, needs validation)
- TX: Forbidden features (leakage risks, target-adjacent)
"""

# ============================================================================
# TIER 0: STRUCTURAL KEYS
# ============================================================================
# These are identifier columns with no predictive value
# Used for joining, grouping, and organizing data

STRUCTURAL_KEYS = [
    # Time identifiers
    'season',
    'week',
    'gameday',
    'game_date',
    'game_time',

    # Team identifiers
    'team',
    'team_home',
    'team_away',
    'opponent',
    'division',

    # Game identifiers
    'game_id',
    'matchup',

    # Location identifiers
    'location',  # 'home' or 'away'
    'home_team',
    'away_team',
]


# ============================================================================
# TIER 1: SAFE CORE FEATURES (T1)
# ============================================================================
# Fundamental team strength metrics - safe to use, well-validated

TEAM_STRENGTH_FEATURES = [
    # EPA metrics (from team_week_epa_2013_2024.csv)
    'off_epa_total',
    'off_epa_per_play',
    'off_success_rate',
    'def_epa_total',
    'def_epa_per_play',
    'def_success_rate',

    # Play volume
    'off_plays',
    'off_pass_plays',
    'off_rush_plays',
    'def_plays',

    # nfelo ratings (from power_ratings_nfelo_*.csv)
    'nfelo',           # Main ELO rating
    'QB Adj',          # QB adjustment to rating
    'Value',           # Overall value rating
    'WoW',             # Week-over-week change
    'YTD',             # Year-to-date performance

    # nfelo efficiency metrics
    'Play',            # Overall play efficiency
    'Pass',            # Pass efficiency
    'Rush',            # Rush efficiency

    # Scoring metrics
    'For',             # Points for
    'Against',         # Points against
    'Dif',             # Point differential
    'Wins',            # Win count
    'Pythag',          # Pythagorean wins
    'Elo',             # Base Elo
    'Film',            # Film grade

    # Substack ratings (from power_ratings_substack_*.csv)
    'Off.',            # Offensive rating
    'Def.',            # Defensive rating
    'Ovr.',            # Overall rating

    # nfelo defensive metrics (suffixed versions from power_ratings_nfelo)
    'def_play_efficiency',   # Defensive play efficiency
    'def_pass_efficiency',   # Defensive pass efficiency
    'def_rush_efficiency',   # Defensive rush efficiency

    # QB performance metrics (from qb_epa_substack)
    'qb_plays',              # QB number of plays
    'qb_value_per_play',     # QB value per play
    'qb_epa_total',          # QB total EPA
    'qb_epa_vs_avg',         # QB EPA vs average
    'qb_epa_vs_replacement', # QB EPA vs replacement

    # Strength of schedule ratings (from strength_of_schedule_nfelo)
    'sos_original_rating',   # Pre-season SOS rating
    'sos_current_rating',    # Current SOS rating
]


# ============================================================================
# TIER 2: MARKET & SITUATIONAL FEATURES (T2)
# ============================================================================
# Context-dependent features that provide edge in specific situations

MARKET_FEATURES = [
    # Rest and schedule
    'rest_days',
    'home_rest_days',
    'away_rest_days',
    'rest_advantage',

    # Situational context
    'div_game',        # Division game indicator
    'is_playoff',      # Playoff game indicator
    'temperature',     # Game temperature
    'wind',            # Wind speed
    'roof',            # Dome/outdoor indicator

    # Team context
    'qb_name',         # Starting QB
    'qb_age',          # QB age
    'qb_change',       # QB change indicator
    'coach_tenure',    # Head coach tenure

    # Forecast metrics (from Substack)
    'Avg. Wins',       # Average projected wins
    'PO%',             # Playoff probability
    'Div%',            # Division win probability
    'Cnf%',            # Conference win probability
    'SB%',             # Super Bowl win probability

    # Win totals and projections
    'projected_wins',      # Projected season wins (from SOS data)
    'projected_win_prob',  # Game-specific win probability

    # Strength of schedule context
    'sos_avg_opp_rating',        # Average opponent rating
    'sos_avg_opp_rating_past',   # Average past opponent rating
    'sos_avg_opp_rating_future', # Average future opponent rating
]


# ============================================================================
# TIER 3: EXPERIMENTAL FEATURES (T3)
# ============================================================================
# Complex derived features that need validation before production use

EXPERIMENTAL_FEATURES = [
    # Rolling window features (ensure leak-free with .shift(1))
    'epa_off_L3',      # 3-game rolling offensive EPA
    'epa_off_L5',      # 5-game rolling offensive EPA
    'epa_off_L10',     # 10-game rolling offensive EPA
    'epa_def_L3',      # 3-game rolling defensive EPA
    'epa_def_L5',      # 5-game rolling defensive EPA
    'epa_def_L10',     # 10-game rolling defensive EPA
    'epa_margin_L3',   # 3-game rolling EPA margin
    'epa_margin_L5',   # 5-game rolling EPA margin
    'epa_margin_L10',  # 10-game rolling EPA margin

    # Recent form
    'win_rate_L3',     # 3-game win rate
    'win_rate_L5',     # 5-game win rate
    'ats_rate_L3',     # 3-game ATS cover rate
    'ats_rate_L5',     # 5-game ATS cover rate
    'point_diff_L3',   # 3-game point differential
    'point_diff_L5',   # 5-game point differential

    # Matchup features
    'home_off_vs_away_def',  # Home offense vs away defense
    'away_off_vs_home_def',  # Away offense vs home defense

    # Interaction terms
    'rest_x_elo_diff',       # Rest advantage × ELO differential
    'qb_adj_x_def_epa',      # QB adjustment × defensive EPA
    'net_play_efficiency',   # Net play efficiency (offense - defense)

    # Ranking features (derived from ratings)
    'sos_rank',              # Overall SOS rank
    'sos_rank_past',         # Past games SOS rank
    'sos_rank_future',       # Future games SOS rank
]


# ============================================================================
# TIER X: FORBIDDEN FEATURES
# ============================================================================
# Features that cause data leakage or are targets themselves

FORBIDDEN_FEATURES = [
    # Vegas lines (these are targets, not features)
    'spread_line',
    'total_line',
    'moneyline_home',
    'moneyline_away',
    'opening_spread',
    'closing_spread',
    'favorite_with_spread',  # Favorite team with spread (contains line)

    # Game outcomes (future information)
    'home_score',
    'away_score',
    'total_score',
    'actual_spread',
    'result',
    'winner',

    # Derived from outcomes
    'home_win',
    'away_win',
    'ats_margin',
    'ats_cover',
    'over_under_result',
]


# ============================================================================
# COMBINED SAFE FEATURES (T0 + T1 + T2)
# ============================================================================
# Features that are safe to use in production models

SAFE_FEATURES = (
    STRUCTURAL_KEYS +
    TEAM_STRENGTH_FEATURES +
    MARKET_FEATURES
)


# ============================================================================
# COLUMN RENAMING MAP
# ============================================================================
# Maps raw column names from data sources to canonical names in tier lists
# Format: {'source_name': 'canonical_name'}

column_name_mapping = {
    # ===== Common mappings across all sources =====
    'Team': 'team',
    'Season': 'season',

    # ===== team_week_epa_2013_2024.csv =====
    # (already uses canonical names, no mapping needed)

    # ===== power_ratings_nfelo_*.csv =====
    'QB Adj': 'QB Adj',    # Keep as-is (in T1)
    'Value': 'Value',      # Keep as-is (in T1)
    'WoW': 'WoW',          # Keep as-is (in T1)
    'YTD': 'YTD',          # Keep as-is (in T1)
    'Play': 'Play',        # Offensive play efficiency (T1)
    'Pass': 'Pass',        # Offensive pass efficiency (T1)
    'Rush': 'Rush',        # Offensive rush efficiency (T1)
    'Play.1': 'def_play_efficiency',  # Defensive play efficiency
    'Pass.1': 'def_pass_efficiency',  # Defensive pass efficiency
    'Rush.1': 'def_rush_efficiency',  # Defensive rush efficiency
    'Play.2': 'net_play_efficiency',  # Net play efficiency (T3)
    'For': 'For',          # Keep as-is (in T1)
    'Against': 'Against',  # Keep as-is (in T1)
    'Dif': 'Dif',          # Keep as-is (in T1)
    'Wins': 'Wins',        # Keep as-is (in T1)
    'Pythag': 'Pythag',    # Keep as-is (in T1)
    'Elo': 'Elo',          # Keep as-is (in T1)
    'Film': 'Film',        # Keep as-is (in T1)
    'nfelo': 'nfelo',      # Keep as-is (in T1)

    # ===== power_ratings_substack_*.csv =====
    # Second row has real column names (after skipping decorative header)
    'Div.': 'division',
    'Off.': 'Off.',        # Keep as-is (in T1)
    'Def.': 'Def.',        # Keep as-is (in T1)
    'Ovr.': 'Ovr.',        # Keep as-is (in T1)
    'Avg. Wins': 'Avg. Wins',  # Keep as-is (in T2)
    'PO%': 'PO%',          # Keep as-is (in T2)
    'Div%': 'Div%',        # Keep as-is (in T2)
    'Cnf%': 'Cnf%',        # Keep as-is (in T2)
    'SB%': 'SB%',          # Keep as-is (in T2)

    # ===== epa_tiers_nfelo_*.csv =====
    'EPA/Play': 'off_epa_per_play',      # Maps to existing T1 feature
    'EPA/Play Against': 'def_epa_per_play',  # Maps to existing T1 feature

    # ===== strength_of_schedule_nfelo_*.csv =====
    'Original Rating': 'sos_original_rating',
    'Current Rating': 'sos_current_rating',
    'Win Total': 'projected_wins',
    'Total Result': 'sos_result',  # Will be ignored (meta)
    'Avg. Opp. Rating': 'sos_avg_opp_rating',  # First occurrence
    'Avg. Opp. Rating.1': 'sos_avg_opp_rating_past',
    'Avg. Opp. Rating.2': 'sos_avg_opp_rating_future',
    'Rank': 'sos_rank',    # First occurrence
    'Rank.1': 'sos_rank_past',
    'Rank.2': 'sos_rank_future',

    # ===== weekly_projections_ppg_substack_*.csv =====
    'Date': 'game_date',
    'Time (ET)': 'game_time',
    'Matchup': 'matchup',
    'Favorite': 'favorite_with_spread',  # TX - forbidden
    'Win Prob.': 'projected_win_prob',

    # ===== qb_epa_substack_*.csv =====
    # Second row has real column names (after skipping decorative header)
    'Player': 'qb_name',
    'Age': 'qb_age',
    'Tms': 'qb_teams',     # Will be ignored (meta)
    'Prim.': 'qb_primary_team',  # Will be ignored (meta)
    'Plays': 'qb_plays',
    'Val/Pl': 'qb_value_per_play',
    'EPA': 'qb_epa_total',
    'vs Avg': 'qb_epa_vs_avg',
    'vs Rep': 'qb_epa_vs_replacement',
}


# ============================================================================
# IGNORED COLUMNS
# ============================================================================
# Columns that exist in raw data but should be explicitly ignored
# (metadata, decorative headers, or low-value features)

ignored_columns = {
    # Metadata columns
    'sos_result',       # Text description (e.g., "Active")
    'qb_teams',         # Number of teams QB played for
    'qb_primary_team',  # Primary team indicator
}


# ============================================================================
# DATASET ROLES
# ============================================================================
# Placeholder for defining the role of each dataset in the pipeline
# Format: {'dataset_name': {'role': str, 'priority': int, 'columns': list}}

dataset_roles = {
    # Historical EPA stats (2013-2024)
    'team_week_epa_2013_2024': {
        'role': 'historical_team_stats',
        'description': 'Historical team EPA metrics by week (2013-2024)',
        'priority': 1,
        'keys': ['season', 'week', 'team'],
        'expected_columns': [
            'season', 'week', 'team',
            'off_epa_total', 'off_epa_per_play', 'off_success_rate',
            'def_epa_total', 'def_epa_per_play', 'def_success_rate',
            'off_plays', 'off_pass_plays', 'off_rush_plays', 'def_plays'
        ],
        'time_range': (2013, 2024),
        'update_frequency': 'historical',
    },

    # Current season power ratings from nfelo
    'power_ratings_nfelo': {
        'role': 'current_power_ratings',
        'description': 'nfelo power ratings for current season',
        'priority': 2,
        'keys': ['season', 'team'],
        'expected_columns': [
            'team', 'season', 'nfelo', 'QB Adj', 'Value', 'WoW', 'YTD',
            'Play', 'Pass', 'Rush', 'def_play_efficiency', 'def_pass_efficiency',
            'def_rush_efficiency', 'net_play_efficiency', 'For', 'Against', 'Dif',
            'Wins', 'Pythag', 'Elo', 'Film'
        ],
        'time_range': (2025, 2025),
        'update_frequency': 'weekly',
        'file_pattern': 'current_season/power_ratings_nfelo_*.csv',
    },

    # Current season power ratings from Substack
    'power_ratings_substack': {
        'role': 'current_power_ratings',
        'description': 'Substack power ratings and forecasts for current season',
        'priority': 2,
        'keys': ['team'],
        'expected_columns': [
            'team', 'division', 'Off.', 'Def.', 'Ovr.',
            'Avg. Wins', 'PO%', 'Div%', 'Cnf%', 'SB%'
        ],
        'time_range': (2025, 2025),
        'update_frequency': 'weekly',
        'file_pattern': 'current_season/power_ratings_substack_*.csv',
        'skip_rows': [0],  # Skip decorative header row
    },

    # EPA tier classifications from nfelo
    'epa_tiers_nfelo': {
        'role': 'current_team_stats',
        'description': 'EPA per play tiers for current season',
        'priority': 3,
        'keys': ['season', 'team'],
        'expected_columns': [
            'team', 'season', 'off_epa_per_play', 'def_epa_per_play'
        ],
        'time_range': (2025, 2025),
        'update_frequency': 'weekly',
        'file_pattern': 'current_season/epa_tiers_nfelo_*.csv',
    },

    # Strength of schedule from nfelo
    'strength_of_schedule_nfelo': {
        'role': 'schedule_context',
        'description': 'Strength of schedule metrics for current season',
        'priority': 3,
        'keys': ['season', 'team'],
        'expected_columns': [
            'team', 'season', 'sos_original_rating', 'sos_current_rating',
            'projected_wins', 'sos_avg_opp_rating', 'sos_avg_opp_rating_past',
            'sos_avg_opp_rating_future', 'sos_rank', 'sos_rank_past', 'sos_rank_future'
        ],
        'time_range': (2025, 2025),
        'update_frequency': 'weekly',
        'file_pattern': 'current_season/strength_of_schedule_nfelo_*.csv',
    },

    # Weekly game projections from Substack
    'weekly_projections_substack': {
        'role': 'game_projections',
        'description': 'Weekly game-by-game projections from Substack',
        'priority': 4,
        'keys': ['game_date', 'matchup'],
        'expected_columns': [
            'game_date', 'game_time', 'matchup',
            'favorite_with_spread', 'projected_win_prob'
        ],
        'time_range': (2025, 2025),
        'update_frequency': 'weekly',
        'file_pattern': 'current_season/weekly_projections_ppg_substack_*.csv',
        # No skip_rows - first row is real header
    },

    # QB EPA stats from Substack
    'qb_epa_substack': {
        'role': 'player_stats',
        'description': 'QB EPA performance metrics from Substack',
        'priority': 3,
        'keys': ['qb_name'],
        'expected_columns': [
            'qb_name', 'qb_age', 'qb_plays', 'qb_value_per_play',
            'qb_epa_total', 'qb_epa_vs_avg', 'qb_epa_vs_replacement'
        ],
        'time_range': (2025, 2025),
        'update_frequency': 'weekly',
        'file_pattern': 'current_season/qb_epa_substack_*.csv',
        'skip_rows': [0],  # Skip decorative header row
    },
}


# ============================================================================
# CANONICAL SCHEMA
# ============================================================================
# Placeholder for the unified v2.0 schema definition
# Format: {'column_name': {'dtype': str, 'description': str, 'tier': str}}

canonical_schema = {
    # =========================================================================
    # T0: STRUCTURAL KEYS
    # =========================================================================
    'season': {
        'dtype': 'int',
        'description': 'NFL season year',
        'tier': 'T0',
        'role': 'id_key',
        'allow_null': False,
    },
    'week': {
        'dtype': 'int',
        'description': 'Week number (1-18 regular season, 19-22 playoffs)',
        'tier': 'T0',
        'role': 'id_key',
        'allow_null': False,
    },
    'team': {
        'dtype': 'str',
        'description': 'Team abbreviation (nfl_data_py standard)',
        'tier': 'T0',
        'role': 'id_key',
        'allow_null': False,
    },
    'game_date': {
        'dtype': 'str',
        'description': 'Game date (YYYY-MM-DD format)',
        'tier': 'T0',
        'role': 'id_key',
        'allow_null': False,
    },
    'game_time': {
        'dtype': 'str',
        'description': 'Game time (ET)',
        'tier': 'T0',
        'role': 'meta_misc',
        'allow_null': True,
    },
    'matchup': {
        'dtype': 'str',
        'description': 'Game matchup description',
        'tier': 'T0',
        'role': 'id_key',
        'allow_null': True,
    },
    'division': {
        'dtype': 'str',
        'description': 'Team division (e.g., NFCW, AFCE)',
        'tier': 'T0',
        'role': 'pre_game_structure',
        'allow_null': True,
    },

    # =========================================================================
    # T1: CORE TEAM STRENGTH FEATURES
    # =========================================================================

    # EPA metrics (from team_week_epa_2013_2024.csv)
    'off_epa_total': {
        'dtype': 'float',
        'description': 'Total offensive EPA',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'off_epa_per_play': {
        'dtype': 'float',
        'description': 'Offensive EPA per play',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'off_success_rate': {
        'dtype': 'float',
        'description': 'Offensive success rate',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'def_epa_total': {
        'dtype': 'float',
        'description': 'Total defensive EPA',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'def_epa_per_play': {
        'dtype': 'float',
        'description': 'Defensive EPA per play',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'def_success_rate': {
        'dtype': 'float',
        'description': 'Defensive success rate',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'off_plays': {
        'dtype': 'int',
        'description': 'Number of offensive plays',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'off_pass_plays': {
        'dtype': 'int',
        'description': 'Number of offensive pass plays',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'off_rush_plays': {
        'dtype': 'int',
        'description': 'Number of offensive rush plays',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'def_plays': {
        'dtype': 'int',
        'description': 'Number of defensive plays',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },

    # nfelo ratings (from power_ratings_nfelo_*.csv)
    'nfelo': {
        'dtype': 'float',
        'description': 'nfelo rating',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'QB Adj': {
        'dtype': 'float',
        'description': 'QB adjustment to rating',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Value': {
        'dtype': 'float',
        'description': 'Overall value rating',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'WoW': {
        'dtype': 'float',
        'description': 'Week-over-week change',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'YTD': {
        'dtype': 'float',
        'description': 'Year-to-date performance',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Play': {
        'dtype': 'float',
        'description': 'Overall play efficiency',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Pass': {
        'dtype': 'float',
        'description': 'Pass efficiency',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Rush': {
        'dtype': 'float',
        'description': 'Rush efficiency',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'For': {
        'dtype': 'float',
        'description': 'Points for',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Against': {
        'dtype': 'float',
        'description': 'Points against',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Dif': {
        'dtype': 'float',
        'description': 'Point differential',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Wins': {
        'dtype': 'int',
        'description': 'Win count',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Pythag': {
        'dtype': 'float',
        'description': 'Pythagorean wins',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Elo': {
        'dtype': 'float',
        'description': 'Base Elo rating',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Film': {
        'dtype': 'float',
        'description': 'Film grade',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'def_play_efficiency': {
        'dtype': 'float',
        'description': 'Defensive play efficiency',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'def_pass_efficiency': {
        'dtype': 'float',
        'description': 'Defensive pass efficiency',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'def_rush_efficiency': {
        'dtype': 'float',
        'description': 'Defensive rush efficiency',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },

    # Substack ratings (from power_ratings_substack_*.csv)
    'Off.': {
        'dtype': 'float',
        'description': 'Offensive rating',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Def.': {
        'dtype': 'float',
        'description': 'Defensive rating',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'Ovr.': {
        'dtype': 'float',
        'description': 'Overall rating',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },

    # QB performance metrics (from qb_epa_substack)
    'qb_plays': {
        'dtype': 'int',
        'description': 'QB number of plays',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'qb_value_per_play': {
        'dtype': 'float',
        'description': 'QB value per play',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'qb_epa_total': {
        'dtype': 'float',
        'description': 'QB total EPA',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'qb_epa_vs_avg': {
        'dtype': 'float',
        'description': 'QB EPA vs average',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'qb_epa_vs_replacement': {
        'dtype': 'float',
        'description': 'QB EPA vs replacement',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },

    # Strength of schedule ratings (from strength_of_schedule_nfelo)
    'sos_original_rating': {
        'dtype': 'float',
        'description': 'Pre-season SOS rating',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },
    'sos_current_rating': {
        'dtype': 'float',
        'description': 'Current SOS rating',
        'tier': 'T1',
        'role': 'pre_game_team_strength',
        'allow_null': True,
        'leakage': 'low',
    },

    # =========================================================================
    # T2: MARKET & SITUATIONAL FEATURES
    # =========================================================================

    'qb_name': {
        'dtype': 'str',
        'description': 'Starting QB name',
        'tier': 'T2',
        'role': 'pre_game_structure',
        'allow_null': True,
        'leakage': 'low',
    },
    'qb_age': {
        'dtype': 'int',
        'description': 'QB age',
        'tier': 'T2',
        'role': 'pre_game_structure',
        'allow_null': True,
        'leakage': 'low',
    },
    'Avg. Wins': {
        'dtype': 'float',
        'description': 'Average projected wins',
        'tier': 'T2',
        'role': 'pre_game_market',
        'allow_null': True,
        'leakage': 'medium',
    },
    'PO%': {
        'dtype': 'float',
        'description': 'Playoff probability',
        'tier': 'T2',
        'role': 'pre_game_market',
        'allow_null': True,
        'leakage': 'medium',
    },
    'Div%': {
        'dtype': 'float',
        'description': 'Division win probability',
        'tier': 'T2',
        'role': 'pre_game_market',
        'allow_null': True,
        'leakage': 'medium',
    },
    'Cnf%': {
        'dtype': 'float',
        'description': 'Conference win probability',
        'tier': 'T2',
        'role': 'pre_game_market',
        'allow_null': True,
        'leakage': 'medium',
    },
    'SB%': {
        'dtype': 'float',
        'description': 'Super Bowl win probability',
        'tier': 'T2',
        'role': 'pre_game_market',
        'allow_null': True,
        'leakage': 'medium',
    },
    'projected_wins': {
        'dtype': 'float',
        'description': 'Projected season wins',
        'tier': 'T2',
        'role': 'pre_game_market',
        'allow_null': True,
        'leakage': 'medium',
    },
    'projected_win_prob': {
        'dtype': 'float',
        'description': 'Game-specific win probability',
        'tier': 'T2',
        'role': 'pre_game_market',
        'allow_null': True,
        'leakage': 'medium',
    },
    'sos_avg_opp_rating': {
        'dtype': 'float',
        'description': 'Average opponent rating',
        'tier': 'T2',
        'role': 'pre_game_structure',
        'allow_null': True,
        'leakage': 'low',
    },
    'sos_avg_opp_rating_past': {
        'dtype': 'float',
        'description': 'Average past opponent rating',
        'tier': 'T2',
        'role': 'pre_game_structure',
        'allow_null': True,
        'leakage': 'low',
    },
    'sos_avg_opp_rating_future': {
        'dtype': 'float',
        'description': 'Average future opponent rating',
        'tier': 'T2',
        'role': 'pre_game_structure',
        'allow_null': True,
        'leakage': 'low',
    },
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_feature_tier(feature_name):
    """
    Determine the tier of a given feature.

    Args:
        feature_name (str): Name of the feature

    Returns:
        str: Tier classification ('T0', 'T1', 'T2', 'T3', 'TX', or 'UNKNOWN')
    """
    if feature_name in STRUCTURAL_KEYS:
        return 'T0'
    elif feature_name in TEAM_STRENGTH_FEATURES:
        return 'T1'
    elif feature_name in MARKET_FEATURES:
        return 'T2'
    elif feature_name in EXPERIMENTAL_FEATURES:
        return 'T3'
    elif feature_name in FORBIDDEN_FEATURES:
        return 'TX'
    else:
        return 'UNKNOWN'


def get_features_by_tier(tier):
    """
    Get all features for a specific tier.

    Args:
        tier (str): Tier name ('T0', 'T1', 'T2', 'T3', 'TX')

    Returns:
        list: Feature names in that tier
    """
    tier_map = {
        'T0': STRUCTURAL_KEYS,
        'T1': TEAM_STRENGTH_FEATURES,
        'T2': MARKET_FEATURES,
        'T3': EXPERIMENTAL_FEATURES,
        'TX': FORBIDDEN_FEATURES,
    }
    return tier_map.get(tier, [])


def is_safe_feature(feature_name):
    """
    Check if a feature is safe to use in production models.

    Args:
        feature_name (str): Name of the feature

    Returns:
        bool: True if safe (T0, T1, or T2), False otherwise
    """
    return feature_name in SAFE_FEATURES


def get_all_tiers_summary():
    """
    Get a summary of all feature tiers.

    Returns:
        dict: Summary statistics for each tier
    """
    return {
        'T0_STRUCTURAL': {
            'count': len(STRUCTURAL_KEYS),
            'description': 'Structural keys and identifiers',
        },
        'T1_CORE': {
            'count': len(TEAM_STRENGTH_FEATURES),
            'description': 'Safe core team strength features',
        },
        'T2_MARKET': {
            'count': len(MARKET_FEATURES),
            'description': 'Market and situational features',
        },
        'T3_EXPERIMENTAL': {
            'count': len(EXPERIMENTAL_FEATURES),
            'description': 'Experimental derived features',
        },
        'TX_FORBIDDEN': {
            'count': len(FORBIDDEN_FEATURES),
            'description': 'Forbidden features (leakage risks)',
        },
        'TOTAL_SAFE': {
            'count': len(SAFE_FEATURES),
            'description': 'Total safe features (T0+T1+T2)',
        },
    }


# ============================================================================
# MODULE INFO
# ============================================================================

__version__ = '2.0.0-alpha'
__author__ = 'Ball Knower v2.0 Team'
__description__ = 'Unified data configuration for Ball Knower v2.0 pipeline'

if __name__ == '__main__':
    # Print tier summary when run directly
    print("\n" + "="*70)
    print("BALL KNOWER v2.0 - FEATURE TIER SUMMARY")
    print("="*70 + "\n")

    summary = get_all_tiers_summary()
    for tier, info in summary.items():
        print(f"{tier:20s}: {info['count']:3d} features - {info['description']}")

    print("\n" + "="*70)
    print(f"Total features defined: {sum(info['count'] for k, info in summary.items() if k != 'TOTAL_SAFE')}")
    print("="*70 + "\n")
