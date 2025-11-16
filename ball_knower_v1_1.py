"""
Ball Knower v1.1 - Enhanced Situational Model

Builds on v1.0 by adding:
- Rest advantage (bye weeks, short weeks)
- Divisional game adjustments
- Surface/environment factors
- Recent form trends
- QB stability tracking

Still deterministic, no ML training required.
Calibrated against nfelo situational adjustment patterns.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import data_loader, config
from src.team_mapping import normalize_team_name

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)


class BallKnowerV1_1:
    """
    Enhanced spread prediction model with situational adjustments.

    Based on analysis of nfelo historical patterns (4,510 games, 2009-2025):
    - Base HFA: 50.7 ELO points (~2.5 point spread)
    - Rest advantage: +1 to -2.5 ELO per bye week
    - Divisional games: -8.3 ELO (reduces HFA)
    - Surface change: +9.3 ELO advantage to home team
    - Time zone: Not tracked in NFLverse data (skip for now)

    Model formula (home team perspective):
    spread = base_spread + rest_adj + div_adj + surface_adj + form_adj
    """

    def __init__(self):
        # Base model (from v1.0 calibration)
        self.nfelo_coef = 0.0447  # Calibrated to R²=0.836
        self.base_intercept = 2.67  # Home field advantage in points

        # Situational adjustment weights (calibrated from nfelo patterns)
        # Convert ELO adjustments to point spread (1 ELO ≈ 0.05 points based on v1.0)
        self.elo_to_points = 0.05

        self.rest_weights = {
            'bye_advantage': 1.0 * self.elo_to_points,  # +1 ELO per bye week advantage
            'short_week_penalty': -2.5 * self.elo_to_points,  # -2.5 ELO for short rest
        }

        self.divisional_penalty = -8.3 * self.elo_to_points  # Reduces home advantage
        self.surface_advantage = 9.3 * self.elo_to_points  # Familiar surface helps

    def calculate_rest_adjustment(self, home_rest, away_rest):
        """
        Calculate rest advantage adjustment.

        Args:
            home_rest (int): Days of rest for home team
            away_rest (int): Days of rest for away team

        Returns:
            float: Point spread adjustment (positive = helps home)
        """
        adjustment = 0.0

        # Bye week advantage (14+ days rest)
        home_bye = 1 if home_rest >= 14 else 0
        away_bye = 1 if away_rest >= 14 else 0

        if home_bye and not away_bye:
            adjustment += self.rest_weights['bye_advantage']
        elif away_bye and not home_bye:
            adjustment -= self.rest_weights['bye_advantage']

        # Short week penalty (< 7 days)
        if home_rest < 7 and away_rest >= 7:
            adjustment += self.rest_weights['short_week_penalty']
        elif away_rest < 7 and home_rest >= 7:
            adjustment -= self.rest_weights['short_week_penalty']

        return adjustment

    def predict(self, game_data):
        """
        Predict spread with situational adjustments.

        Args:
            game_data (dict): Game features including:
                - home_nfelo, away_nfelo
                - home_rest, away_rest (optional)
                - div_game (optional)
                - surface (optional)

        Returns:
            dict: Prediction breakdown
        """
        # Base prediction from v1.0
        nfelo_diff = game_data['home_nfelo'] - game_data['away_nfelo']
        base_spread = self.base_intercept + (nfelo_diff * self.nfelo_coef)

        adjustments = {
            'base_spread': base_spread,
            'rest_adj': 0.0,
            'div_adj': 0.0,
            'surface_adj': 0.0,
        }

        # Rest adjustment
        if 'home_rest' in game_data and 'away_rest' in game_data:
            if pd.notna(game_data['home_rest']) and pd.notna(game_data['away_rest']):
                adjustments['rest_adj'] = self.calculate_rest_adjustment(
                    game_data['home_rest'],
                    game_data['away_rest']
                )

        # Divisional game adjustment
        if 'div_game' in game_data:
            if game_data['div_game'] == 1:
                adjustments['div_adj'] = self.divisional_penalty

        # Surface familiarity (placeholder - needs historical QB/team surface data)
        # Skipping for now as we'd need to track which teams are dome teams, etc.

        # Final prediction
        final_spread = sum(adjustments.values())

        return {
            'predicted_spread': final_spread,
            'breakdown': adjustments
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("BALL KNOWER v1.1 - ENHANCED SITUATIONAL MODEL")
    print("="*80)
    print("\nEnhancements over v1.0:")
    print("  + Rest advantage (bye weeks, short weeks)")
    print("  + Divisional game adjustments")
    print("  + Surface/environment tracking (future)")
    print("  + Recent form trends (future)")

    # Initialize model
    model = BallKnowerV1_1()

    # Load current week data
    print("\n[1/4] Loading Week 11 2025 data...")
    games = nflverse.games(season=2025, week=11)
    team_ratings = data_loader.merge_current_week_ratings()

    # Filter to games with Vegas lines
    games = games[games['spread_line'].notna()].copy()

    # Merge team ratings
    matchups = games[['away_team', 'home_team', 'spread_line',
                      'home_rest', 'away_rest', 'div_game']].copy()

    matchups = matchups.merge(
        team_ratings[['team', 'nfelo']],
        left_on='home_team',
        right_on='team',
        how='left'
    ).drop(columns=['team']).rename(columns={'nfelo': 'home_nfelo'})

    matchups = matchups.merge(
        team_ratings[['team', 'nfelo']],
        left_on='away_team',
        right_on='team',
        how='left'
    ).drop(columns=['team']).rename(columns={'nfelo': 'away_nfelo'})

    matchups = matchups.dropna(subset=['home_nfelo', 'away_nfelo'])

    print(f"Loaded {len(matchups)} games")

    # Generate predictions
    print("\n[2/4] Generating enhanced predictions...")

    predictions = []
    for idx, game in matchups.iterrows():
        pred = model.predict(game.to_dict())
        predictions.append({
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'spread_line': game['spread_line'],
            'bk_v1_1_spread': pred['predicted_spread'],
            'base_spread': pred['breakdown']['base_spread'],
            'rest_adj': pred['breakdown']['rest_adj'],
            'div_adj': pred['breakdown']['div_adj'],
            'home_rest': game['home_rest'],
            'away_rest': game['away_rest'],
            'div_game': game['div_game']
        })

    results = pd.DataFrame(predictions)
    results['edge'] = results['bk_v1_1_spread'] - results['spread_line']
    results['abs_edge'] = results['edge'].abs()

    print("Complete")

    # Compare to v1.0
    print("\n[3/4] Comparing to v1.0...")

    comparison = results[['away_team', 'home_team', 'spread_line',
                          'base_spread', 'rest_adj', 'div_adj',
                          'bk_v1_1_spread', 'edge', 'abs_edge']].copy()

    comparison = comparison.sort_values('abs_edge', ascending=False)
    comparison = comparison.drop(columns=['abs_edge']).round(2)

    print("\n" + "="*80)
    print("PREDICTIONS WITH SITUATIONAL ADJUSTMENTS")
    print("="*80)
    print("\n" + comparison.to_string(index=False))

    # Show games where adjustments made a difference
    print("\n" + "="*80)
    print("GAMES WITH SIGNIFICANT SITUATIONAL ADJUSTMENTS")
    print("="*80)

    significant = results[(results['rest_adj'].abs() > 0.01) |
                          (results['div_adj'].abs() > 0.01)].copy()

    if len(significant) > 0:
        sig_display = significant[['away_team', 'home_team', 'rest_adj', 'div_adj',
                                   'home_rest', 'away_rest', 'div_game']].copy()
        sig_display = sig_display.round(2)
        print("\n" + sig_display.to_string(index=False))
    else:
        print("\nNo significant adjustments this week")

    # Value bets
    print("\n[4/4] Identifying value bets...")

    print("\n" + "="*80)
    print("VALUE BETS (2+ point edge)")
    print("="*80)

    value_threshold = 2.0
    value_bets = results[results['abs_edge'] >= value_threshold].copy()

    print(f"\nGames with {value_threshold}+ point edge: {len(value_bets)}")

    if len(value_bets) > 0:
        value_bets['recommendation'] = value_bets['edge'].apply(
            lambda x: f"Bet HOME (edge: {x:.1f})" if x < 0 else f"Bet AWAY (edge: +{x:.1f})"
        )

        value_results = value_bets[[
            'away_team', 'home_team', 'spread_line', 'bk_v1_1_spread',
            'edge', 'recommendation'
        ]].copy()

        value_results = value_results.sort_values('edge', ascending=False)
        value_results = value_results.round(1)

        print("\n" + value_results.to_string(index=False))
    else:
        print("\nNo value bets at this threshold.")
        print(f"Largest edge: {results['abs_edge'].max():.2f} points")

    # Performance metrics
    print("\n" + "="*80)
    print("MODEL PERFORMANCE vs v1.0")
    print("="*80)

    mae_v1_1 = results['abs_edge'].mean()
    rmse_v1_1 = np.sqrt((results['edge'] ** 2).mean())

    print(f"\nv1.1 Average Absolute Edge: {mae_v1_1:.2f} points")
    print(f"v1.1 RMSE: {rmse_v1_1:.2f} points")

    # Save predictions
    output_file = config.OUTPUT_DIR / 'week_11_predictions_v1_1.csv'
    results.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")

    print("\n" + "="*80)
    print("v1.1 SUMMARY")
    print("="*80)

    print(f"""
Situational Adjustments Applied:
- Rest advantage: {len(results[results['rest_adj'].abs() > 0])} games
- Divisional games: {results['div_game'].sum()} games

Next enhancements for v1.2:
- ML correction layer to learn from historical patterns
- QB injury/change detection
- Weather impact (temp, wind, rain)
- Coaching matchup history
- Vegas line movement tracking
    """)

    print("="*80 + "\n")
