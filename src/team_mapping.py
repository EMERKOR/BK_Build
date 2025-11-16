"""
Team Name Normalization Module

This module handles all team name variations across different data sources:
- nfl_data_py (standard): ARI, ATL, BAL, BUF, etc.
- nfelo: LAR, DET, KC, etc.
- Substack power ratings: Rams, Seahawks, Chiefs, etc.
- Substack QB EPA: buf, kan, ram (lowercase)
- Substack weekly projections: Los Angeles Rams, New England Patriots, etc.

All team names are normalized to nfl_data_py standard abbreviations.
"""

# Standard nfl_data_py team abbreviations (2025 season)
NFL_DATA_PY_TEAMS = [
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
    'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
    'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
    'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
]

# Comprehensive team name mapping to standard abbreviations
TEAM_NAME_MAPPING = {
    # Arizona Cardinals
    'ARI': 'ARI', 'Arizona': 'ARI', 'Cardinals': 'ARI', 'Arizona Cardinals': 'ARI',
    'crd': 'ARI', 'ari': 'ARI',

    # Atlanta Falcons
    'ATL': 'ATL', 'Atlanta': 'ATL', 'Falcons': 'ATL', 'Atlanta Falcons': 'ATL',
    'atl': 'ATL',

    # Baltimore Ravens
    'BAL': 'BAL', 'Baltimore': 'BAL', 'Ravens': 'BAL', 'Baltimore Ravens': 'BAL',
    'rav': 'BAL', 'bal': 'BAL',

    # Buffalo Bills
    'BUF': 'BUF', 'Buffalo': 'BUF', 'Bills': 'BUF', 'Buffalo Bills': 'BUF',
    'buf': 'BUF',

    # Carolina Panthers
    'CAR': 'CAR', 'Carolina': 'CAR', 'Panthers': 'CAR', 'Carolina Panthers': 'CAR',
    'car': 'CAR',

    # Chicago Bears
    'CHI': 'CHI', 'Chicago': 'CHI', 'Bears': 'CHI', 'Chicago Bears': 'CHI',
    'chi': 'CHI',

    # Cincinnati Bengals
    'CIN': 'CIN', 'Cincinnati': 'CIN', 'Bengals': 'CIN', 'Cincinnati Bengals': 'CIN',
    'cin': 'CIN',

    # Cleveland Browns
    'CLE': 'CLE', 'Cleveland': 'CLE', 'Browns': 'CLE', 'Cleveland Browns': 'CLE',
    'cle': 'CLE',

    # Dallas Cowboys
    'DAL': 'DAL', 'Dallas': 'DAL', 'Cowboys': 'DAL', 'Dallas Cowboys': 'DAL',
    'dal': 'DAL',

    # Denver Broncos
    'DEN': 'DEN', 'Denver': 'DEN', 'Broncos': 'DEN', 'Denver Broncos': 'DEN',
    'den': 'DEN',

    # Detroit Lions
    'DET': 'DET', 'Detroit': 'DET', 'Lions': 'DET', 'Detroit Lions': 'DET',
    'det': 'DET',

    # Green Bay Packers
    'GB': 'GB', 'GNB': 'GB', 'Green Bay': 'GB', 'Packers': 'GB', 'Green Bay Packers': 'GB',
    'gnb': 'GB', 'gb': 'GB',

    # Houston Texans
    'HOU': 'HOU', 'HTX': 'HOU', 'Houston': 'HOU', 'Texans': 'HOU', 'Houston Texans': 'HOU',
    'htx': 'HOU', 'hou': 'HOU',

    # Indianapolis Colts
    'IND': 'IND', 'CLT': 'IND', 'Indianapolis': 'IND', 'Colts': 'IND', 'Indianapolis Colts': 'IND',
    'clt': 'IND', 'ind': 'IND',

    # Jacksonville Jaguars
    'JAX': 'JAX', 'JAC': 'JAX', 'Jacksonville': 'JAX', 'Jaguars': 'JAX', 'Jacksonville Jaguars': 'JAX',
    'jax': 'JAX', 'jac': 'JAX',

    # Kansas City Chiefs
    'KC': 'KC', 'KAN': 'KC', 'Kansas City': 'KC', 'Chiefs': 'KC', 'Kansas City Chiefs': 'KC',
    'kan': 'KC', 'kc': 'KC',

    # Las Vegas Raiders (formerly Oakland)
    'LV': 'LV', 'OAK': 'LV', 'RAI': 'LV', 'Las Vegas': 'LV', 'Raiders': 'LV',
    'Las Vegas Raiders': 'LV', 'Oakland Raiders': 'LV', 'Oakland': 'LV',
    'rai': 'LV', 'oak': 'LV', 'lv': 'LV',

    # Los Angeles Chargers (formerly San Diego)
    'LAC': 'LAC', 'SDG': 'LAC', 'SD': 'LAC', 'Los Angeles Chargers': 'LAC', 'LA Chargers': 'LAC',
    'Chargers': 'LAC', 'San Diego Chargers': 'LAC',
    'sdg': 'LAC', 'lac': 'LAC',

    # Los Angeles Rams (formerly St. Louis)
    'LAR': 'LAR', 'STL': 'LAR', 'LA Rams': 'LAR', 'Los Angeles Rams': 'LAR',
    'St. Louis Rams': 'LAR', 'St Louis Rams': 'LAR', 'Rams': 'LAR',
    'ram': 'LAR', 'lar': 'LAR', 'stl': 'LAR',

    # Miami Dolphins
    'MIA': 'MIA', 'Miami': 'MIA', 'Dolphins': 'MIA', 'Miami Dolphins': 'MIA',
    'mia': 'MIA',

    # Minnesota Vikings
    'MIN': 'MIN', 'Minnesota': 'MIN', 'Vikings': 'MIN', 'Minnesota Vikings': 'MIN',
    'min': 'MIN',

    # New England Patriots
    'NE': 'NE', 'NWE': 'NE', 'New England': 'NE', 'Patriots': 'NE', 'New England Patriots': 'NE',
    'nwe': 'NE', 'ne': 'NE',

    # New Orleans Saints
    'NO': 'NO', 'NOR': 'NO', 'New Orleans': 'NO', 'Saints': 'NO', 'New Orleans Saints': 'NO',
    'nor': 'NO', 'no': 'NO',

    # New York Giants
    'NYG': 'NYG', 'New York Giants': 'NYG', 'NY Giants': 'NYG', 'Giants': 'NYG',
    'nyg': 'NYG',

    # New York Jets
    'NYJ': 'NYJ', 'New York Jets': 'NYJ', 'NY Jets': 'NYJ', 'Jets': 'NYJ',
    'nyj': 'NYJ',

    # Philadelphia Eagles
    'PHI': 'PHI', 'Philadelphia': 'PHI', 'Eagles': 'PHI', 'Philadelphia Eagles': 'PHI',
    'phi': 'PHI',

    # Pittsburgh Steelers
    'PIT': 'PIT', 'Pittsburgh': 'PIT', 'Steelers': 'PIT', 'Pittsburgh Steelers': 'PIT',
    'pit': 'PIT',

    # San Francisco 49ers
    'SF': 'SF', 'SFO': 'SF', 'San Francisco': 'SF', '49ers': 'SF', 'San Francisco 49ers': 'SF',
    'sfo': 'SF', 'sf': 'SF',

    # Seattle Seahawks
    'SEA': 'SEA', 'Seattle': 'SEA', 'Seahawks': 'SEA', 'Seattle Seahawks': 'SEA',
    'sea': 'SEA',

    # Tampa Bay Buccaneers
    'TB': 'TB', 'TAM': 'TB', 'Tampa Bay': 'TB', 'Buccaneers': 'TB', 'Tampa Bay Buccaneers': 'TB',
    'Bucs': 'TB',
    'tam': 'TB', 'tb': 'TB',

    # Tennessee Titans
    'TEN': 'TEN', 'OTI': 'TEN', 'Tennessee': 'TEN', 'Titans': 'TEN', 'Tennessee Titans': 'TEN',
    'oti': 'TEN', 'ten': 'TEN',

    # Washington Commanders (formerly Football Team, Redskins)
    'WAS': 'WAS', 'WSH': 'WAS', 'Washington': 'WAS', 'Commanders': 'WAS',
    'Washington Commanders': 'WAS', 'Washington Football Team': 'WAS',
    'was': 'WAS', 'wsh': 'WAS',
}


def normalize_team_name(team_name):
    """
    Normalize any team name variation to nfl_data_py standard abbreviation.

    Args:
        team_name (str): Team name in any format

    Returns:
        str: Standardized team abbreviation (e.g., 'KC', 'BUF', 'LAR')

    Raises:
        ValueError: If team name cannot be mapped
    """
    if not isinstance(team_name, str):
        raise ValueError(f"Team name must be string, got {type(team_name)}")

    # Strip whitespace
    team_name = team_name.strip()

    # Try direct lookup
    if team_name in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[team_name]

    # Try case-insensitive lookup
    for key, value in TEAM_NAME_MAPPING.items():
        if key.lower() == team_name.lower():
            return value

    raise ValueError(f"Unknown team name: '{team_name}'. Cannot map to standard abbreviation.")


def normalize_team_column(df, column_name='team', new_column_name='team_std'):
    """
    Add a standardized team column to a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with team names
        column_name (str): Name of column containing team names
        new_column_name (str): Name of new standardized column

    Returns:
        pd.DataFrame: DataFrame with new standardized team column
    """
    import pandas as pd

    df = df.copy()
    df[new_column_name] = df[column_name].apply(normalize_team_name)
    return df


# Quick reference for common variations
NFELO_TO_STD = {
    'LAR': 'LAR', 'DET': 'DET', 'KC': 'KC', 'SEA': 'SEA', 'PHI': 'PHI',
    'BAL': 'BAL', 'BUF': 'BUF', 'GB': 'GB', 'IND': 'IND', 'DEN': 'DEN',
    'HOU': 'HOU', 'LAC': 'LAC', 'NE': 'NE', 'TB': 'TB', 'SF': 'SF',
    'CHI': 'CHI', 'PIT': 'PIT', 'DAL': 'DAL', 'MIN': 'MIN', 'ATL': 'ATL',
    'JAX': 'JAX', 'ARI': 'ARI', 'WAS': 'WAS', 'NYG': 'NYG', 'CIN': 'CIN',
    'NYJ': 'NYJ', 'CLE': 'CLE', 'MIA': 'MIA', 'CAR': 'CAR', 'OAK': 'LV',
    'NO': 'NO', 'TEN': 'TEN',
    # Historical relocations
    'STL': 'LAR', 'SD': 'LAC', 'SDG': 'LAC'
}

SUBSTACK_NICKNAME_TO_STD = {
    'Rams': 'LAR', 'Lions': 'DET', 'Chiefs': 'KC', 'Seahawks': 'SEA',
    'Eagles': 'PHI', 'Ravens': 'BAL', 'Bills': 'BUF', 'Packers': 'GB',
    'Colts': 'IND', 'Broncos': 'DEN', 'Texans': 'HOU', 'Chargers': 'LAC',
    'Patriots': 'NE', 'Buccaneers': 'TB', '49ers': 'SF', 'Bears': 'CHI',
    'Steelers': 'PIT', 'Cowboys': 'DAL', 'Vikings': 'MIN', 'Falcons': 'ATL',
    'Jaguars': 'JAX', 'Cardinals': 'ARI', 'Commanders': 'WAS', 'Giants': 'NYG',
    'Bengals': 'CIN', 'Jets': 'NYJ', 'Browns': 'CLE', 'Dolphins': 'MIA',
    'Panthers': 'CAR', 'Raiders': 'LV', 'Saints': 'NO', 'Titans': 'TEN'
}
