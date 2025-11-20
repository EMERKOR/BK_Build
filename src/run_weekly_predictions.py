"""
Ball Knower Weekly Predictions CLI

Generate weekly NFL predictions using calibrated weights and unified data loaders.

Usage:
    python src/run_weekly_predictions.py --season 2025 --week 11
    python src/run_weekly_predictions.py --season 2025 --week 11 --output my_predictions.csv

Output CSV columns:
    - game_id: Unique game identifier
    - season: NFL season year
    - week: Week number
    - away_team: Away team abbreviation
    - home_team: Home team abbreviation
    - bk_line: Ball Knower predicted spread (negative = home favored)
    - vegas_line: Closing spread from data source
    - edge: Difference between BK and Vegas (bk_line - vegas_line)
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import Ball Knower modules
from ball_knower.io import loaders
from ball_knower.modeling import models
from ball_knower.features import engineering as features
from src import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate weekly NFL predictions using Ball Knower models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/run_weekly_predictions.py --season 2025 --week 11
  python src/run_weekly_predictions.py --season 2025 --week 12 --output predictions_week12.csv
        """
    )

    parser.add_argument(
        '--season',
        type=int,
        required=True,
        help='NFL season year (e.g., 2025)'
    )

    parser.add_argument(
        '--week',
        type=int,
        required=True,
        help='Week number (1-18 for regular season)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: output/predictions_{season}_week_{week}.csv)'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['v1.0', 'v1.1'],
        default='v1.1',
        help='Model version to use (default: v1.1 - EnhancedSpreadModel with calibrated weights)'
    )

    return parser.parse_args()


def load_weekly_data(season, week):
    """
    Load all required data for the given season and week.

    Args:
        season (int): NFL season year
        week (int): Week number

    Returns:
        tuple: (team_ratings, matchups) DataFrames
    """
    print(f"\n[1/4] Loading data for {season} Week {week}...")

    # Load all data sources using unified loaders
    all_data = loaders.load_all_sources(season=season, week=week)

    # Get merged team ratings
    team_ratings = all_data['merged_ratings']
    print(f"  Loaded ratings for {len(team_ratings)} teams")

    # Load weekly projections to get matchups
    weekly_projections = loaders.load_weekly_projections_ppg("substack", season=season, week=week)

    # Parse matchups
    if 'team_away' in weekly_projections.columns and 'team_home' in weekly_projections.columns:
        matchups = weekly_projections[['team_away', 'team_home']].copy()

        # Extract Vegas spread if available
        if 'Favorite' in weekly_projections.columns:
            matchups['vegas_line'] = weekly_projections['Favorite'].str.extract(r'([-+]?\d+\.?\d*)')[0].astype(float)
        else:
            matchups['vegas_line'] = np.nan
    else:
        # Fallback: parse from Matchup column
        from src.team_mapping import normalize_team_name
        weekly_projections['team_away'] = weekly_projections['Matchup'].str.split(' at | vs ').str[0].apply(normalize_team_name)
        weekly_projections['team_home'] = weekly_projections['Matchup'].str.split(' at | vs ').str[1].apply(normalize_team_name)

        matchups = weekly_projections[['team_away', 'team_home']].copy()

        if 'Favorite' in weekly_projections.columns:
            matchups['vegas_line'] = weekly_projections['Favorite'].str.extract(r'([-+]?\d+\.?\d*)')[0].astype(float)
        else:
            matchups['vegas_line'] = np.nan

    print(f"  Found {len(matchups)} scheduled games")

    return team_ratings, matchups


def build_feature_matrix(matchups, team_ratings):
    """
    Build feature matrix by merging home/away team ratings with matchups.

    Uses unified prepare_inference_features() from ball_knower.features.engineering.

    Args:
        matchups (pd.DataFrame): Game matchups with team_away, team_home
        team_ratings (pd.DataFrame): Team ratings with normalized 'team' column

    Returns:
        pd.DataFrame: Merged feature matrix with home/away features
    """
    print("\n[2/4] Building feature matrix...")

    # Use unified feature preparation from ball_knower
    feature_df = features.prepare_inference_features(matchups, team_ratings)

    print(f"  Feature matrix shape: {feature_df.shape}")

    return feature_df


def generate_predictions(feature_df, model_version='v1.1'):
    """
    Generate predictions for all games using the specified model.

    Args:
        feature_df (pd.DataFrame): Feature matrix with home/away features
        model_version (str): Model version ('v1.0' or 'v1.1')

    Returns:
        list: Prediction dictionaries
    """
    print(f"\n[3/4] Generating predictions with {model_version} model...")

    # Instantiate model (will automatically load calibrated weights)
    if model_version == 'v1.0':
        model = models.DeterministicSpreadModel(use_calibrated=True)
        model_name = 'v1.0'
    else:
        model = models.EnhancedSpreadModel(use_calibrated=True)
        model_name = 'v1.1'

    predictions = []

    for idx, game in feature_df.iterrows():
        # Extract home team features
        home_features = {
            'nfelo': game.get('nfelo_home'),
            'epa_margin': game.get('epa_margin_home'),
            'Ovr.': game.get('Ovr._home'),
            'rest_days': game.get('rest_days_home'),
            'win_rate_L5': game.get('win_rate_L5_home'),
            'QB Adj': game.get('QB Adj_home')
        }

        # Extract away team features
        away_features = {
            'nfelo': game.get('nfelo_away'),
            'epa_margin': game.get('epa_margin_away'),
            'Ovr.': game.get('Ovr._away'),
            'rest_days': game.get('rest_days_away'),
            'win_rate_L5': game.get('win_rate_L5_away'),
            'QB Adj': game.get('QB Adj_away')
        }

        # Generate prediction
        bk_line = model.predict(home_features, away_features)

        # Calculate edge (only if vegas_line is available)
        vegas_line = game.get('vegas_line')
        if pd.notna(vegas_line):
            edge = bk_line - vegas_line
        else:
            edge = np.nan

        # Create game_id
        game_id = f"{game['season']}_{game['week']:02d}_{game['team_away']}_{game['team_home']}"

        predictions.append({
            'game_id': game_id,
            'season': game['season'],
            'week': game['week'],
            'away_team': game['team_away'],
            'home_team': game['team_home'],
            'bk_line': round(bk_line, 1),
            'vegas_line': vegas_line if pd.notna(vegas_line) else None,
            'edge': round(edge, 1) if pd.notna(edge) else None
        })

    print(f"  Generated {len(predictions)} predictions")

    return predictions


def save_predictions(predictions, season, week, output_path=None):
    """
    Save predictions to CSV file.

    Args:
        predictions (list): List of prediction dictionaries
        season (int): Season year
        week (int): Week number
        output_path (str): Custom output path (optional)
    """
    print("\n[4/4] Saving predictions...")

    # Create DataFrame
    df = pd.DataFrame(predictions)

    # Determine output path
    if output_path is None:
        output_dir = config.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"predictions_{season}_week_{week}.csv"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"  Saved {len(df)} predictions to: {output_file}")

    return df, output_file


def print_summary(predictions_df):
    """Print summary of predictions."""
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)

    print(f"\nTotal games: {len(predictions_df)}")

    # Show predictions sorted by absolute edge
    if 'edge' in predictions_df.columns and predictions_df['edge'].notna().any():
        print("\nTop predictions by edge (absolute value):")
        print("\nSpread Convention: Negative = Home Favored")
        print("Edge = BK Line - Vegas Line\n")

        top_edges = predictions_df.dropna(subset=['edge']).copy()
        top_edges['abs_edge'] = top_edges['edge'].abs()
        top_edges = top_edges.sort_values('abs_edge', ascending=False).head(10)

        display_cols = ['away_team', 'home_team', 'vegas_line', 'bk_line', 'edge']
        print(top_edges[display_cols].to_string(index=False))

        # Value bets
        value_threshold = config.MIN_BET_EDGE
        value_bets = predictions_df[predictions_df['edge'].abs() >= value_threshold]

        if len(value_bets) > 0:
            print(f"\n" + "="*80)
            print(f"VALUE BETS (Edge >= {value_threshold} pts)")
            print("="*80)

            value_bets = value_bets.copy()
            value_bets['recommendation'] = value_bets.apply(
                lambda row: f"Bet {row['home_team']}" if row['edge'] < 0 else f"Bet {row['away_team']}",
                axis=1
            )

            display_cols = ['away_team', 'home_team', 'vegas_line', 'bk_line', 'edge', 'recommendation']
            print("\n" + value_bets[display_cols].sort_values('edge', key=abs, ascending=False).to_string(index=False))
            print(f"\nFound {len(value_bets)} value bets")
        else:
            print(f"\nNo value bets found with edge >= {value_threshold} pts")
    else:
        print("\nAll predictions:")
        display_cols = ['away_team', 'home_team', 'bk_line']
        print("\n" + predictions_df[display_cols].to_string(index=False))

    print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    print("\n" + "="*80)
    print(f"BALL KNOWER - WEEKLY PREDICTIONS")
    print("="*80)
    print(f"\nSeason: {args.season}")
    print(f"Week: {args.week}")
    print(f"Model: {args.model}")

    try:
        # Load data
        team_ratings, matchups = load_weekly_data(args.season, args.week)

        # Add season/week to matchups for feature matrix
        matchups['season'] = args.season
        matchups['week'] = args.week

        # Build feature matrix
        feature_df = build_feature_matrix(matchups, team_ratings)

        # Generate predictions
        predictions = generate_predictions(feature_df, model_version=args.model)

        # Save predictions
        predictions_df, output_file = save_predictions(
            predictions,
            args.season,
            args.week,
            args.output
        )

        # Print summary
        print_summary(predictions_df)

        print(f"✓ Success! Predictions saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
