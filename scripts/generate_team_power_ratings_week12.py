#!/usr/bin/env python

"""
Generate blended Ball_Knower team power ratings for a given week.

Inputs:
  - Inpredict market ratings (manual dict in this script for now)
  - The Athletic power rankings (manual ranks)
  - PFF power rankings (manual ranks)
  - nfelo team power ratings CSV
  - Substack team ratings: PPG (Ovr.) + Elo forecast CSVs

Output:
  - subjective/team_power_ratings_YYYY_weekNN.csv
"""

import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

SEASON = 2025
WEEK = 12

# Weights for each pillar (must not necessarily sum to 1.0, but they do here)
MARKET_WEIGHT     = 0.40
OBJECTIVE_WEIGHT  = 0.30
ANALYST_WEIGHT    = 0.20
STRUCTURAL_WEIGHT = 0.00   # reserved for Ben Baldwin / structural edge pillar
SUBJECTIVE_WEIGHT = 0.10   # health / vibes / form sliders

# How much to scale Substack z-scores to "points"
SUBSTACK_POINTS_SD = 3.0

# File paths (relative to repo root; tweak as needed)
NFelo_PATH      = Path("data/Team Season Power Ratings-export-2025-11-20.csv")
SUBSTACK_PPG    = Path("data/data-w9KQp (1).csv")   # 2025 Forecast sheet w/ Ovr.
SUBSTACK_ELO    = Path("data/data-ZZZAZ.csv")       # Elo forecast sheet

OUTPUT_PATH     = Path("subjective/team_power_ratings_2025_week12.csv")

# --------------------------------------------------------------------
# MANUAL DATA (market + rankings + team code maps)
# --------------------------------------------------------------------

market_ratings_map = {
    'LAR': 6.4, 'KC': 5.9, 'BUF': 5.5, 'DET': 5.4, 'SEA': 5.2, 'PHI': 5.0,
    'GB': 4.3, 'IND': 3.6, 'BAL': 3.4,
    'NE': 2.4, 'LAC': 1.8, 'SF': 1.7, 'DEN': 1.6, 'TB': 1.4,
    'JAX': 0.3, 'DAL': -0.2, 'MIN': -0.7, 'CHI': -1.2, 'HOU': -1.5, 'PIT': -2.0,
    'ARI': -3.3, 'MIA': -3.3, 'NYG': -3.8, 'CAR': -4.3, 'LV': -5.0,
    'ATL': -5.8, 'NO': -6.0, 'CLE': -6.1, 'CIN': -6.3, 'WAS': -6.8,
    'NYJ': -8.7, 'TEN': -8.7,
}

athletic_ranks_map = {
    'PHI': 1, 'LAR': 2, 'IND': 3, 'DEN': 4, 'NE': 5, 'SEA': 6, 'BUF': 7, 'DET': 8,
    'TB': 9, 'GB': 10, 'SF': 11, 'CHI': 12, 'PIT': 13, 'JAX': 14, 'LAC': 15,
    'KC': 16, 'BAL': 17, 'CAR': 18, 'HOU': 19, 'DAL': 20, 'MIN': 21, 'ARI': 22,
    'CIN': 23, 'MIA': 24, 'ATL': 25, 'WAS': 26, 'NO': 27, 'NYG': 28, 'NYJ': 29,
    'LV': 30, 'CLE': 31, 'TEN': 32,
}

pff_ranks_map = {
    'LAR': 1, 'KC': 2, 'BAL': 3, 'GB': 4, 'BUF': 5, 'DET': 6, 'PHI': 7, 'SEA': 8,
    'DEN': 9, 'SF': 10, 'IND': 11, 'LAC': 12, 'HOU': 13, 'TB': 14, 'NE': 15,
    'CHI': 16, 'JAX': 17, 'MIN': 18, 'PIT': 19, 'DAL': 20, 'MIA': 21, 'ARI': 22,
    'CAR': 23, 'ATL': 24, 'WAS': 25, 'CIN': 26, 'LV': 27, 'NYG': 28, 'CLE': 29,
    'NYJ': 30, 'NO': 31, 'TEN': 32,
}

code_to_name = {
    'LAR': 'Rams', 'KC': 'Chiefs', 'BUF': 'Bills', 'DET': 'Lions', 'SEA': 'Seahawks',
    'PHI': 'Eagles', 'GB': 'Packers', 'IND': 'Colts', 'BAL': 'Ravens',
    'NE': 'Patriots', 'LAC': 'Chargers', 'SF': '49ers', 'DEN': 'Broncos',
    'TB': 'Buccaneers', 'JAX': 'Jaguars', 'DAL': 'Cowboys', 'MIN': 'Vikings',
    'CHI': 'Bears', 'HOU': 'Texans', 'PIT': 'Steelers', 'ARI': 'Cardinals',
    'MIA': 'Dolphins', 'NYG': 'Giants', 'CAR': 'Panthers', 'LV': 'Raiders',
    'ATL': 'Falcons', 'NO': 'Saints', 'CLE': 'Browns', 'CIN': 'Bengals',
    'WAS': 'Commanders', 'NYJ': 'Jets', 'TEN': 'Titans',
}

team_name_to_code = {v: k for k, v in code_to_name.items()}


# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------

def z_score(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def rank_to_rating(rank: int) -> float:
    """
    Map rank 1–32 linearly to +3 to -3.
    """
    return 3.0 - (rank - 1) * (6.0 / 31.0)


# --------------------------------------------------------------------
# LOADERS
# --------------------------------------------------------------------

def load_nfelo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[['Team', 'Value']].rename(columns={'Team': 'team_code', 'Value': 'nfelo_value'})


def load_substack_composite(ppg_path: Path, elo_path: Path) -> pd.DataFrame:
    # PPG / "Ovr." ratings (header row starts at index 1)
    ppg = pd.read_csv(ppg_path, header=1)
    ppg = ppg[['Team', 'Ovr.']].rename(columns={'Ovr.': 'substack_ppg_raw'})
    ppg['team_code'] = ppg['Team'].map(team_name_to_code)

    # Elo forecast (header row starts at index 1)
    elo = pd.read_csv(elo_path, header=1)
    elo = elo[['Team', 'Elo']].rename(columns={'Elo': 'substack_elo_raw'})
    elo['team_code'] = elo['Team'].map(team_name_to_code)

    merged = pd.merge(ppg, elo, on='team_code', suffixes=('_ppg', '_elo'))

    merged['z_ppg'] = z_score(merged['substack_ppg_raw'])
    merged['z_elo'] = z_score(merged['substack_elo_raw'])
    merged['substack_z'] = 0.5 * merged['z_ppg'] + 0.5 * merged['z_elo']

    # Put on a rough "points" scale
    merged['substack_points'] = merged['substack_z'] * SUBSTACK_POINTS_SD

    return merged[['team_code', 'substack_points']]


# --------------------------------------------------------------------
# MAIN BUILD
# --------------------------------------------------------------------

def build_weekly_blended_ratings() -> pd.DataFrame:
    teams = list(market_ratings_map.keys())
    df = pd.DataFrame({'team_code': teams})

    # Manual pillars
    df['market_rating'] = df['team_code'].map(market_ratings_map)
    df['athletic_rank'] = df['team_code'].map(athletic_ranks_map)
    df['pff_rank'] = df['team_code'].map(pff_ranks_map)

    # Objective data
    nfelo_df = load_nfelo(NFelo_PATH)
    substack_df = load_substack_composite(SUBSTACK_PPG, SUBSTACK_ELO)

    df = df.merge(nfelo_df, on='team_code', how='left')
    df = df.merge(substack_df, on='team_code', how='left')

    df = df.fillna(0.0)

    # Analyst pillar
    df['athletic_rating'] = df['athletic_rank'].apply(rank_to_rating)
    df['pff_rating'] = df['pff_rank'].apply(rank_to_rating)
    df['analyst_composite_rating'] = 0.5 * df['athletic_rating'] + 0.5 * df['pff_rating']

    # Objective pillar (points-scale)
    df['objective_composite'] = 0.5 * df['nfelo_value'] + 0.5 * df['substack_points']

    # Structural pillar placeholder (will be fed from Baldwin-style CSV later)
    df['structural_edge'] = 0.0

    # Subjective pillar placeholders (hooks for your sliders / notes)
    df['subjective_health_adj'] = 0.0
    df['subjective_form_adj'] = 0.0
    df['subjective_total'] = df['subjective_health_adj'] + df['subjective_form_adj']

    # Final blended rating
    df['bk_blended_rating'] = (
        MARKET_WEIGHT     * df['market_rating'] +
        OBJECTIVE_WEIGHT  * df['objective_composite'] +
        ANALYST_WEIGHT    * df['analyst_composite_rating'] +
        STRUCTURAL_WEIGHT * df['structural_edge'] +
        SUBJECTIVE_WEIGHT * df['subjective_total']
    )

    # Add metadata / names
    df['season'] = SEASON
    df['week'] = WEEK
    df['team_name'] = df['team_code'].map(code_to_name)

    # Column order
    cols = [
        'season', 'week', 'team_code', 'team_name',
        'market_rating',
        'nfelo_value', 'substack_points', 'objective_composite',
        'athletic_rank', 'pff_rank', 'analyst_composite_rating',
        'structural_edge',
        'subjective_health_adj', 'subjective_form_adj', 'subjective_total',
        'bk_blended_rating',
    ]
    df = df[cols].sort_values('bk_blended_rating', ascending=False)

    return df


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ratings = build_weekly_blended_ratings()
    ratings.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved blended ratings to {OUTPUT_PATH}")
    print(ratings.head().to_string(index=False))


if __name__ == "__main__":
    main()
