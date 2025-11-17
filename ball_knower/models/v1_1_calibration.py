"""
Ball Knower v1.1 - Calibrated Spread Model

This module provides calibrated weights for the deterministic spread model
by fitting to historical Vegas lines using ordinary least squares regression.

The v1.0 model uses fixed weights:
    - nfelo_diff: 0.02
    - substack_power_diff: 0.5
    - epa_off_diff: 35.0
    - epa_def_diff: -35.0 (defensive EPA, better = lower)

The v1.1 model learns optimal weights from historical data by solving:
    vegas_line ≈ w_nfelo * nfelo_diff + w_substack * substack_diff +
                 w_epa_off * epa_off_diff + w_epa_def * epa_def_diff + bias

All predictions are from the HOME TEAM perspective:
    - Negative spread = home team favored
    - Positive spread = home team underdog
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
import warnings

from ball_knower.io import loaders


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_schedule_data(season: int, data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load NFL schedule data with Vegas lines.

    Args:
        season: NFL season year
        data_dir: Optional directory containing schedule data (defaults to repo/data/cache)

    Returns:
        DataFrame with game schedule and Vegas lines
    """
    if data_dir is None:
        # Default to repo_root/data/cache
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "data" / "cache"
    else:
        data_dir = Path(data_dir)

    schedule_file = data_dir / f"schedules_{season}.csv"

    if not schedule_file.exists():
        raise FileNotFoundError(
            f"Schedule file not found: {schedule_file}. "
            f"Please ensure schedules_{season}.csv exists in {data_dir}"
        )

    schedule = pd.read_csv(schedule_file)

    # Filter for regular season only
    if 'game_type' in schedule.columns:
        schedule = schedule[schedule['game_type'] == 'REG'].copy()

    # Keep only games with valid spread lines
    schedule = schedule[schedule['spread_line'].notna()].copy()

    return schedule


def prepare_training_matrix(
    season: int,
    weeks: List[int],
    data_dir: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare training matrix X and target vector y for weight calibration.

    Loads historical weeks and builds a feature matrix with model components:
        - nfelo_diff: nfelo home - nfelo away
        - substack_power_diff: Ovr. home - Ovr. away
        - epa_off_diff: epa_off home - epa_off away
        - epa_def_diff: epa_def home - epa_def away

    Target y is the Vegas spread line (negative = home favored).

    Args:
        season: NFL season year
        weeks: List of week numbers to include in training
        data_dir: Optional directory for data files (defaults to repo/data/current_season)

    Returns:
        Tuple of:
            - X: Feature matrix (N x 4) - [nfelo_diff, substack_diff, epa_off_diff, epa_def_diff]
            - y: Target vector (N,) - Vegas spread lines
            - games_df: DataFrame with game metadata and features

    Example:
        >>> X, y, games = prepare_training_matrix(2025, [1, 2, 3, 4, 5])
        >>> print(f"Training on {len(games)} games")
        >>> print(f"Feature matrix shape: {X.shape}")
    """
    # Load schedule to get all games
    schedule = load_schedule_data(season)

    # Filter for requested weeks
    schedule = schedule[schedule['week'].isin(weeks)].copy()

    if len(schedule) == 0:
        raise ValueError(f"No games found for season {season}, weeks {weeks}")

    print(f"Loading data for {len(schedule)} games across weeks {min(weeks)}-{max(weeks)}")

    # Set default data directory if not provided
    if data_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "data" / "current_season"

    # Collect all features for each game
    games_list = []

    for week in sorted(weeks):
        week_games = schedule[schedule['week'] == week].copy()

        if len(week_games) == 0:
            continue

        # Try to load team ratings for this week
        try:
            # Load all rating sources
            nfelo_power = loaders.load_power_ratings("nfelo", season, week, data_dir)
            nfelo_epa = loaders.load_epa_tiers("nfelo", season, week, data_dir)
            substack_power = loaders.load_power_ratings("substack", season, week, data_dir)

            # Rename EPA columns if needed
            if 'EPA/Play' in nfelo_epa.columns:
                nfelo_epa = nfelo_epa.rename(columns={
                    'EPA/Play': 'epa_off',
                    'EPA/Play Against': 'epa_def'
                })

            # Add epa_margin if not present
            if 'epa_margin' not in nfelo_epa.columns:
                nfelo_epa['epa_margin'] = nfelo_epa['epa_off'] - nfelo_epa['epa_def']

            # Merge ratings
            ratings = nfelo_power[['team', 'nfelo']].copy()
            ratings = ratings.merge(
                nfelo_epa[['team', 'epa_off', 'epa_def']],
                on='team',
                how='left'
            )
            ratings = ratings.merge(
                substack_power[['team', 'Ovr.']],
                on='team',
                how='left'
            )

            # For each game, extract features
            for idx, game in week_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']

                # Get home and away ratings
                home_ratings = ratings[ratings['team'] == home_team]
                away_ratings = ratings[ratings['team'] == away_team]

                if len(home_ratings) == 0 or len(away_ratings) == 0:
                    warnings.warn(
                        f"Missing ratings for {away_team} @ {home_team} (Week {week}). Skipping.",
                        UserWarning
                    )
                    continue

                home_ratings = home_ratings.iloc[0]
                away_ratings = away_ratings.iloc[0]

                # Calculate differentials (home - away)
                nfelo_diff = home_ratings['nfelo'] - away_ratings['nfelo']
                substack_diff = home_ratings['Ovr.'] - away_ratings['Ovr.']
                epa_off_diff = home_ratings['epa_off'] - away_ratings['epa_off']
                epa_def_diff = home_ratings['epa_def'] - away_ratings['epa_def']

                # Store game with features
                games_list.append({
                    'season': season,
                    'week': week,
                    'game_id': game.get('game_id', f"{season}_{week:02d}_{away_team}_{home_team}"),
                    'away_team': away_team,
                    'home_team': home_team,
                    'vegas_line': game['spread_line'],
                    'nfelo_diff': nfelo_diff,
                    'substack_power_diff': substack_diff,
                    'epa_off_diff': epa_off_diff,
                    'epa_def_diff': epa_def_diff,
                    'home_score': game.get('home_score', None),
                    'away_score': game.get('away_score', None),
                })

        except FileNotFoundError as e:
            warnings.warn(
                f"Could not load ratings for week {week}: {e}. Skipping week.",
                UserWarning
            )
            continue

    # Convert to DataFrame
    games_df = pd.DataFrame(games_list)

    if len(games_df) == 0:
        raise ValueError(f"No valid games found with complete ratings for weeks {weeks}")

    # Build feature matrix X and target vector y
    feature_cols = ['nfelo_diff', 'substack_power_diff', 'epa_off_diff', 'epa_def_diff']
    X = games_df[feature_cols].values
    y = games_df['vegas_line'].values

    print(f"✓ Prepared training matrix: {X.shape[0]} games, {X.shape[1]} features")
    print(f"  Features: {feature_cols}")
    print(f"  Target: vegas_line")

    return X, y, games_df


# ============================================================================
# WEIGHT CALIBRATION
# ============================================================================

def calibrate_weights(
    season: int,
    weeks: List[int],
    data_dir: Optional[Union[str, Path]] = None
) -> Dict[str, float]:
    """
    Calibrate model weights using ordinary least squares regression.

    Solves the equation: vegas_line ≈ X @ w + bias
    where X contains [nfelo_diff, substack_diff, epa_off_diff, epa_def_diff]

    Uses numpy.linalg.lstsq to find optimal weights that minimize
    the squared error between predicted and actual Vegas lines.

    Args:
        season: NFL season year
        weeks: List of week numbers to train on
        data_dir: Optional directory for data files

    Returns:
        Dictionary with calibrated weights:
            - weight_nfelo: Weight for nfelo differential
            - weight_substack: Weight for Substack power differential
            - weight_epa_off: Weight for offensive EPA differential
            - weight_epa_def: Weight for defensive EPA differential
            - bias: Intercept term (should be close to 0 if features are symmetric)

    Example:
        >>> weights = calibrate_weights(2025, list(range(1, 11)))
        >>> print(f"nfelo weight: {weights['weight_nfelo']:.4f}")
        >>> print(f"Substack weight: {weights['weight_substack']:.4f}")
    """
    # Prepare training data
    X, y, games_df = prepare_training_matrix(season, weeks, data_dir)

    # Add bias term (column of ones)
    X_with_bias = np.column_stack([X, np.ones(len(X))])

    # Solve least squares: y = X_with_bias @ [w, bias]
    # numpy.linalg.lstsq returns (solution, residuals, rank, singular_values)
    solution, residuals, rank, s = np.linalg.lstsq(X_with_bias, y, rcond=None)

    # Extract weights and bias
    weights = {
        'weight_nfelo': solution[0],
        'weight_substack': solution[1],
        'weight_epa_off': solution[2],
        'weight_epa_def': solution[3],
        'bias': solution[4],
    }

    # Calculate training metrics
    y_pred = X_with_bias @ solution
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    print("\n" + "="*70)
    print("CALIBRATED WEIGHTS (v1.1)")
    print("="*70)
    print(f"  nfelo weight:     {weights['weight_nfelo']:>8.4f}")
    print(f"  substack weight:  {weights['weight_substack']:>8.4f}")
    print(f"  epa_off weight:   {weights['weight_epa_off']:>8.4f}")
    print(f"  epa_def weight:   {weights['weight_epa_def']:>8.4f}")
    print(f"  bias:             {weights['bias']:>8.4f}")
    print("="*70)
    print(f"Training Performance:")
    print(f"  MAE:  {mae:.3f} points")
    print(f"  RMSE: {rmse:.3f} points")
    print(f"  Games: {len(games_df)}")
    print("="*70 + "\n")

    return weights


# ============================================================================
# PREDICTION WITH CALIBRATED WEIGHTS
# ============================================================================

def build_week_lines_v1_1(
    season: int,
    week: int,
    weights: Dict[str, float],
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Build spread predictions for a week using calibrated v1.1 weights.

    Also includes v1.0 predictions (with fixed weights) for comparison.

    Args:
        season: NFL season year
        week: NFL week number
        weights: Calibrated weights from calibrate_weights()
        data_dir: Optional directory for data files

    Returns:
        DataFrame with columns:
            - away_team, home_team: Team names
            - vegas_line: Vegas spread line (if available)
            - bk_line_v1_1: Ball Knower v1.1 prediction (calibrated weights)
            - bk_line_v1_0: Ball Knower v1.0 prediction (fixed weights)
            - edge_v1_1: v1.1 edge over Vegas
            - edge_v1_0: v1.0 edge over Vegas
            - nfelo_diff, substack_power_diff, epa_off_diff, epa_def_diff: Components

    Example:
        >>> weights = calibrate_weights(2025, list(range(1, 11)))
        >>> lines = build_week_lines_v1_1(2025, 11, weights)
        >>> print(lines[['away_team', 'home_team', 'vegas_line', 'bk_line_v1_1', 'edge_v1_1']])
    """
    # Set default data directory if not provided
    if data_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "data" / "current_season"

    # Load team ratings
    nfelo_power = loaders.load_power_ratings("nfelo", season, week, data_dir)
    nfelo_epa = loaders.load_epa_tiers("nfelo", season, week, data_dir)
    substack_power = loaders.load_power_ratings("substack", season, week, data_dir)

    # Rename EPA columns if needed
    if 'EPA/Play' in nfelo_epa.columns:
        nfelo_epa = nfelo_epa.rename(columns={
            'EPA/Play': 'epa_off',
            'EPA/Play Against': 'epa_def'
        })

    # Add epa_margin if not present
    if 'epa_margin' not in nfelo_epa.columns:
        nfelo_epa['epa_margin'] = nfelo_epa['epa_off'] - nfelo_epa['epa_def']

    # Merge ratings
    ratings = nfelo_power[['team', 'nfelo']].copy()
    ratings = ratings.merge(
        nfelo_epa[['team', 'epa_off', 'epa_def']],
        on='team',
        how='left'
    )
    ratings = ratings.merge(
        substack_power[['team', 'Ovr.']],
        on='team',
        how='left'
    )

    # Load weekly matchups (contains Vegas lines)
    matchups = loaders.load_weekly_projections_ppg("substack", season, week, data_dir)

    # Parse matchups if needed
    if 'team_away' not in matchups.columns or 'team_home' not in matchups.columns:
        # Try to parse from Matchup column
        if 'Matchup' in matchups.columns:
            from ball_knower.io.loaders import normalize_team

            def parse_matchup(matchup):
                if ' at ' in matchup:
                    teams = matchup.split(' at ')
                    return pd.Series({
                        'team_away': normalize_team(teams[0]),
                        'team_home': normalize_team(teams[1])
                    })
                elif ' vs ' in matchup:
                    teams = matchup.split(' vs ')
                    return pd.Series({
                        'team_away': normalize_team(teams[0]),
                        'team_home': normalize_team(teams[1])
                    })
                else:
                    return pd.Series({'team_away': None, 'team_home': None})

            matchups[['team_away', 'team_home']] = matchups['Matchup'].apply(parse_matchup)
        else:
            raise ValueError("Cannot determine matchups from data. Need 'team_away'/'team_home' or 'Matchup' column.")

    # Extract Vegas lines if available
    if 'Favorite' in matchups.columns and 'substack_spread_line' not in matchups.columns:
        matchups['substack_spread_line'] = matchups['Favorite'].str.extract(r'([-+]?\d+\.?\d*)')[0].astype(float)

    # Build predictions for each game
    predictions = []

    for idx, game in matchups.iterrows():
        home_team = game['team_home']
        away_team = game['team_away']

        if pd.isna(home_team) or pd.isna(away_team):
            continue

        # Get ratings
        home_ratings = ratings[ratings['team'] == home_team]
        away_ratings = ratings[ratings['team'] == away_team]

        if len(home_ratings) == 0 or len(away_ratings) == 0:
            warnings.warn(
                f"Missing ratings for {away_team} @ {home_team}. Skipping.",
                UserWarning
            )
            continue

        home_ratings = home_ratings.iloc[0]
        away_ratings = away_ratings.iloc[0]

        # Calculate differentials
        nfelo_diff = home_ratings['nfelo'] - away_ratings['nfelo']
        substack_diff = home_ratings['Ovr.'] - away_ratings['Ovr.']
        epa_off_diff = home_ratings['epa_off'] - away_ratings['epa_off']
        epa_def_diff = home_ratings['epa_def'] - away_ratings['epa_def']

        # v1.1 prediction with calibrated weights
        bk_line_v1_1 = (
            weights['weight_nfelo'] * nfelo_diff +
            weights['weight_substack'] * substack_diff +
            weights['weight_epa_off'] * epa_off_diff +
            weights['weight_epa_def'] * epa_def_diff +
            weights['bias']
        )

        # v1.0 prediction with fixed weights (for comparison)
        # v1.0: nfelo*0.02, substack*0.5, epa_margin*35 (where epa_margin = epa_off - epa_def)
        # We'll use: nfelo*0.02, substack*0.5, epa_off*35, epa_def*-35
        bk_line_v1_0 = (
            0.02 * nfelo_diff +
            0.5 * substack_diff +
            35.0 * epa_off_diff +
            -35.0 * epa_def_diff
        )

        # Get Vegas line
        vegas_line = game.get('substack_spread_line', None)

        # Calculate edges
        edge_v1_1 = bk_line_v1_1 - vegas_line if vegas_line is not None else None
        edge_v1_0 = bk_line_v1_0 - vegas_line if vegas_line is not None else None

        predictions.append({
            'away_team': away_team,
            'home_team': home_team,
            'vegas_line': vegas_line,
            'bk_line_v1_1': round(bk_line_v1_1, 1),
            'bk_line_v1_0': round(bk_line_v1_0, 1),
            'edge_v1_1': round(edge_v1_1, 1) if edge_v1_1 is not None else None,
            'edge_v1_0': round(edge_v1_0, 1) if edge_v1_0 is not None else None,
            'nfelo_diff': round(nfelo_diff, 2),
            'substack_power_diff': round(substack_diff, 2),
            'epa_off_diff': round(epa_off_diff, 3),
            'epa_def_diff': round(epa_def_diff, 3),
        })

    predictions_df = pd.DataFrame(predictions)

    print(f"✓ Generated predictions for {len(predictions_df)} games (Week {week})")

    return predictions_df
