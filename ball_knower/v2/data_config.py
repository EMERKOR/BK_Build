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

    # Team identifiers
    'team',
    'team_home',
    'team_away',
    'opponent',

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
    'qb_change',       # QB change indicator
    'coach_tenure',    # Head coach tenure

    # Forecast metrics (from Substack)
    'Avg. Wins',       # Average projected wins
    'PO%',             # Playoff probability
    'Div%',            # Division win probability
    'Cnf%',            # Conference win probability
    'SB%',             # Super Bowl win probability
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
# Placeholder for standardizing column names across data sources
# Format: {'source_name': 'canonical_name'}

column_name_mapping = {
    # nfelo → canonical
    'Team': 'team',
    'Season': 'season',

    # Substack → canonical
    'Off.': 'substack_off',
    'Def.': 'substack_def',
    'Ovr.': 'substack_ovr',

    # Add more mappings as sources are integrated
}


# ============================================================================
# DATASET ROLES
# ============================================================================
# Placeholder for defining the role of each dataset in the pipeline
# Format: {'dataset_name': {'role': str, 'priority': int, 'columns': list}}

dataset_roles = {
    'team_week_epa_2013_2024': {
        'role': 'historical_stats',
        'priority': 1,
        'columns': [
            'season', 'week', 'team',
            'off_epa_total', 'off_epa_per_play', 'off_success_rate',
            'def_epa_total', 'def_epa_per_play', 'def_success_rate',
            'off_plays', 'off_pass_plays', 'off_rush_plays', 'def_plays'
        ],
        'time_range': (2013, 2024),
    },

    'power_ratings_nfelo': {
        'role': 'current_ratings',
        'priority': 2,
        'columns': [
            'Team', 'Season', 'nfelo', 'QB Adj', 'Value', 'WoW', 'YTD',
            'Play', 'Pass', 'Rush', 'For', 'Against', 'Dif',
            'Wins', 'Pythag', 'Elo', 'Film'
        ],
        'time_range': (2025, 2025),
    },

    'power_ratings_substack': {
        'role': 'current_ratings',
        'priority': 2,
        'columns': ['Team', 'Off.', 'Def.', 'Ovr.', 'Avg. Wins', 'PO%', 'Div%', 'Cnf%', 'SB%'],
        'time_range': (2025, 2025),
    },

    # Placeholder for additional datasets
}


# ============================================================================
# CANONICAL SCHEMA
# ============================================================================
# Placeholder for the unified v2.0 schema definition
# Format: {'column_name': {'dtype': str, 'description': str, 'tier': str}}

canonical_schema = {
    # T0: Structural Keys
    'season': {
        'dtype': 'int',
        'description': 'NFL season year',
        'tier': 'T0',
        'required': True,
    },
    'week': {
        'dtype': 'int',
        'description': 'Week number (1-18 regular season, 19-22 playoffs)',
        'tier': 'T0',
        'required': True,
    },
    'team': {
        'dtype': 'str',
        'description': 'Team abbreviation (nfl_data_py standard)',
        'tier': 'T0',
        'required': True,
    },

    # T1: Core Features (examples)
    'off_epa_per_play': {
        'dtype': 'float',
        'description': 'Offensive EPA per play',
        'tier': 'T1',
        'required': False,
        'source': 'team_week_epa_2013_2024',
    },
    'def_epa_per_play': {
        'dtype': 'float',
        'description': 'Defensive EPA per play',
        'tier': 'T1',
        'required': False,
        'source': 'team_week_epa_2013_2024',
    },
    'nfelo': {
        'dtype': 'float',
        'description': 'nfelo rating',
        'tier': 'T1',
        'required': False,
        'source': 'power_ratings_nfelo',
    },

    # Add more schema definitions as pipeline develops
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
