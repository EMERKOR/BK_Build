"""
Comprehensive Feature Engineering - Ball Knower v2.0

Builds rich feature set from ALL available data sources:
- Team EPA (offense/defense, pass/rush)
- Player stats (QB, WR, RB, OL, DL, secondary)
- Injuries (who's out, backup quality)
- Weather (wind, temp, precipitation)
- Referees (scoring tendencies)
- Matchups (specific unit vs unit edges)
- Advanced metrics (PFR, NGS, FTN)

Philosophy:
- Wind, refs, injuries are FEATURES, not standalone betting signals
- Train on actual outcomes (not Vegas lines)
- Only bet when comprehensive model shows edge
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class ComprehensiveFeatureBuilder:
    """
    Builds comprehensive feature set for NFL game prediction

    Combines data from multiple sources into a single feature vector
    per game that captures team quality, player performance, context,
    and matchup-specific edges.
    """

    def __init__(self, data_dir: str = '.'):
        """
        Initialize feature builder with data directory

        Args:
            data_dir: Path to directory containing parquet files
        """
        self.data_dir = Path(data_dir)
        self.data_cache = {}

    def load_all_data(self):
        """Load all data sources into memory"""
        print("Loading all data sources...")

        # Core data
        self.schedules = pd.read_parquet(self.data_dir / 'schedules.parquet')
        self.team_stats = pd.read_parquet(self.data_dir / 'team_stats_week.parquet')

        # EPA data (aggregated)
        self.team_epa = pd.read_csv(self.data_dir / 'team_week_epa_2013_2024.csv')

        # Player-level data
        self.player_stats = pd.read_parquet(self.data_dir / 'player_stats_week.parquet')
        self.rosters = pd.read_parquet(self.data_dir / 'rosters_weekly.parquet')
        self.snap_counts = pd.read_parquet(self.data_dir / 'snap_counts.parquet')

        # Injury data
        from team_mapping import normalize_team_name
        self.injuries = pd.read_parquet(self.data_dir / 'injuries.parquet')
        self.injuries['team'] = self.injuries['team'].apply(normalize_team_name)

        # Advanced stats
        self.pfr_passing = pd.read_parquet(self.data_dir / 'pfr_adv_pass_week.parquet')
        self.pfr_rushing = pd.read_parquet(self.data_dir / 'pfr_adv_rush_week.parquet')
        self.pfr_receiving = pd.read_parquet(self.data_dir / 'pfr_adv_rec_week.parquet')
        self.pfr_defense = pd.read_parquet(self.data_dir / 'pfr_adv_def_week.parquet')

        # NGS stats
        self.ngs_passing = pd.read_parquet(self.data_dir / 'ngs_passing.parquet')
        self.ngs_rushing = pd.read_parquet(self.data_dir / 'ngs_rushing.parquet')
        self.ngs_receiving = pd.read_parquet(self.data_dir / 'ngs_receiving.parquet')

        print("✓ All data loaded")

    def build_game_features(self,
                           season: int,
                           week: int,
                           home_team: str,
                           away_team: str) -> Dict:
        """
        Build comprehensive feature set for a single game

        Args:
            season: Season year
            week: Week number
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Dictionary of features
        """
        features = {}

        # 1. Team-level EPA features
        features.update(self._build_team_epa_features(
            season, week, home_team, away_team
        ))

        # 2. Player performance features
        features.update(self._build_player_features(
            season, week, home_team, away_team
        ))

        # 3. Injury impact features
        features.update(self._build_injury_features(
            season, week, home_team, away_team
        ))

        # 4. Context features (weather, referee)
        features.update(self._build_context_features(
            season, week, home_team, away_team
        ))

        # 5. Matchup-specific features
        features.update(self._build_matchup_features(
            season, week, home_team, away_team
        ))

        return features

    def _build_team_epa_features(self, season, week, home_team, away_team) -> Dict:
        """
        Team-level EPA features from team_epa CSV

        Features:
        - Offensive EPA per play
        - Defensive EPA per play
        - Recent form (last 3/5 games)
        - Season-to-date averages
        """
        features = {}

        # Get recent games for both teams (lookback window to avoid leakage)
        home_recent = self.team_epa[
            (self.team_epa['season'] == season) &
            (self.team_epa['team'] == home_team) &
            (self.team_epa['week'] < week)
        ].sort_values('week', ascending=False).head(5)

        away_recent = self.team_epa[
            (self.team_epa['season'] == season) &
            (self.team_epa['team'] == away_team) &
            (self.team_epa['week'] < week)
        ].sort_values('week', ascending=False).head(5)

        # Home team offensive EPA
        if len(home_recent) > 0:
            features['home_off_epa_mean'] = home_recent['off_epa_per_play'].mean()
            features['home_off_epa_recent3'] = home_recent.head(3)['off_epa_per_play'].mean()
        else:
            features['home_off_epa_mean'] = 0.0
            features['home_off_epa_recent3'] = 0.0

        # Away team offensive EPA
        if len(away_recent) > 0:
            features['away_off_epa_mean'] = away_recent['off_epa_per_play'].mean()
            features['away_off_epa_recent3'] = away_recent.head(3)['off_epa_per_play'].mean()
        else:
            features['away_off_epa_mean'] = 0.0
            features['away_off_epa_recent3'] = 0.0

        # Home team defensive EPA (lower is better)
        if len(home_recent) > 0:
            features['home_def_epa_mean'] = home_recent['def_epa_per_play'].mean()
        else:
            features['home_def_epa_mean'] = 0.0

        # Away team defensive EPA
        if len(away_recent) > 0:
            features['away_def_epa_mean'] = away_recent['def_epa_per_play'].mean()
        else:
            features['away_def_epa_mean'] = 0.0

        return features

    def _build_player_features(self, season, week, home_team, away_team) -> Dict:
        """
        Player-level performance features

        Features:
        - QB efficiency (recent games)
        - WR production
        - RB effectiveness
        - Pass protection quality
        - Pass rush quality
        """
        features = {}

        # Get starting QBs from rosters
        home_qb_stats = self._get_qb_stats(season, week, home_team)
        away_qb_stats = self._get_qb_stats(season, week, away_team)

        features['home_qb_rating'] = home_qb_stats.get('passer_rating', 85.0)
        features['away_qb_rating'] = away_qb_stats.get('passer_rating', 85.0)

        features['home_qb_completion_pct'] = home_qb_stats.get('completion_pct', 0.60)
        features['away_qb_completion_pct'] = away_qb_stats.get('completion_pct', 0.60)

        # TODO: Add WR, RB, OL, DL stats when we build those extractors

        return features

    def _build_injury_features(self, season, week, home_team, away_team) -> Dict:
        """
        Injury impact features

        Features:
        - Key injuries (QB, WR1, LT, etc.)
        - Number of starters out
        - Backup quality differential
        """
        features = {}

        # Get injuries for this week
        home_injuries = self.injuries[
            (self.injuries['season'] == season) &
            (self.injuries['week'] == week) &
            (self.injuries['team'] == home_team)
        ]

        away_injuries = self.injuries[
            (self.injuries['season'] == season) &
            (self.injuries['week'] == week) &
            (self.injuries['team'] == away_team)
        ]

        # Count key position injuries
        features['home_qb_out'] = int(
            ((home_injuries['position'] == 'QB') &
             (home_injuries['report_status'] == 'Out')).any()
        )
        features['away_qb_out'] = int(
            ((away_injuries['position'] == 'QB') &
             (away_injuries['report_status'] == 'Out')).any()
        )

        # Count total "Out" players
        features['home_players_out'] = (home_injuries['report_status'] == 'Out').sum()
        features['away_players_out'] = (away_injuries['report_status'] == 'Out').sum()

        return features

    def _build_context_features(self, season, week, home_team, away_team) -> Dict:
        """
        Contextual features (weather, referee, rest, etc.)

        Features:
        - Wind speed
        - Temperature
        - Precipitation
        - Referee scoring tendency
        - Days rest
        - Division game
        """
        features = {}

        # Get game from schedule
        game = self.schedules[
            (self.schedules['season'] == season) &
            (self.schedules['week'] == week) &
            (self.schedules['home_team'] == home_team) &
            (self.schedules['away_team'] == away_team)
        ]

        if len(game) > 0:
            game = game.iloc[0]

            # Weather
            features['wind'] = game.get('wind', 5.0)  # Default calm
            features['temp'] = game.get('temp', 60.0)  # Default moderate
            features['is_outdoor'] = int(game.get('roof', 'outdoors') in ['outdoors', 'outdoor'])

            # Referee (encode as historical scoring tendency)
            referee = game.get('referee', 'Unknown')
            features['referee_scoring_tendency'] = self._get_referee_tendency(referee)

            # Rest
            features['home_rest'] = game.get('home_rest', 7)
            features['away_rest'] = game.get('away_rest', 7)

            # Division game
            features['div_game'] = int(game.get('div_game', 0))
        else:
            # Defaults if game not found
            features['wind'] = 5.0
            features['temp'] = 60.0
            features['is_outdoor'] = 1
            features['referee_scoring_tendency'] = 0.0
            features['home_rest'] = 7
            features['away_rest'] = 7
            features['div_game'] = 0

        return features

    def _build_matchup_features(self, season, week, home_team, away_team) -> Dict:
        """
        Matchup-specific features

        Features:
        - Home pass offense vs away pass defense
        - Away pass offense vs home pass defense
        - Home rush offense vs away rush defense
        - Away rush offense vs home rush defense
        """
        features = {}

        # Get team stats for both teams from team_stats (has passing_epa, rushing_epa)
        home_stats = self.team_stats[
            (self.team_stats['season'] == season) &
            (self.team_stats['team'] == home_team) &
            (self.team_stats['week'] < week)
        ].tail(5)

        away_stats = self.team_stats[
            (self.team_stats['season'] == season) &
            (self.team_stats['team'] == away_team) &
            (self.team_stats['week'] < week)
        ].tail(5)

        # Get defensive stats by looking at opponent's offensive performance
        # (defensive EPA = opponent's offensive EPA when playing you)
        home_def_stats = self.team_stats[
            (self.team_stats['season'] == season) &
            (self.team_stats['opponent_team'] == home_team) &
            (self.team_stats['week'] < week)
        ].tail(5)

        away_def_stats = self.team_stats[
            (self.team_stats['season'] == season) &
            (self.team_stats['opponent_team'] == away_team) &
            (self.team_stats['week'] < week)
        ].tail(5)

        if len(home_stats) > 0 and len(away_stats) > 0:
            # Matchup: Home pass off vs Away pass def
            home_pass_off = home_stats['passing_epa'].mean()
            away_pass_def = away_def_stats['passing_epa'].mean() if len(away_def_stats) > 0 else 0.0
            features['home_pass_vs_away_passdef'] = home_pass_off - away_pass_def

            # Matchup: Away pass off vs Home pass def
            away_pass_off = away_stats['passing_epa'].mean()
            home_pass_def = home_def_stats['passing_epa'].mean() if len(home_def_stats) > 0 else 0.0
            features['away_pass_vs_home_passdef'] = away_pass_off - home_pass_def

            # Matchup: Home rush off vs Away rush def
            home_rush_off = home_stats['rushing_epa'].mean()
            away_rush_def = away_def_stats['rushing_epa'].mean() if len(away_def_stats) > 0 else 0.0
            features['home_rush_vs_away_rushdef'] = home_rush_off - away_rush_def

            # Matchup: Away rush off vs Home rush def
            away_rush_off = away_stats['rushing_epa'].mean()
            home_rush_def = home_def_stats['rushing_epa'].mean() if len(home_def_stats) > 0 else 0.0
            features['away_rush_vs_home_rushdef'] = away_rush_off - home_rush_def
        else:
            features['home_pass_vs_away_passdef'] = 0.0
            features['away_pass_vs_home_passdef'] = 0.0
            features['home_rush_vs_away_rushdef'] = 0.0
            features['away_rush_vs_home_rushdef'] = 0.0

        return features

    def _get_qb_stats(self, season, week, team) -> Dict:
        """Get QB stats for a team (recent games average)"""
        # Get QB from recent games
        qb_stats = self.player_stats[
            (self.player_stats['season'] == season) &
            (self.player_stats['team'] == team) &
            (self.player_stats['position'] == 'QB') &
            (self.player_stats['week'] < week)
        ].sort_values('week', ascending=False).head(3)

        if len(qb_stats) > 0:
            # Use passing EPA as proxy for rating (scaled to 0-150 range similar to passer rating)
            avg_epa = qb_stats['passing_epa'].mean()
            scaled_rating = 85.0 + (avg_epa * 10)  # Convert EPA to rating-like scale

            return {
                'passer_rating': scaled_rating,
                'completion_pct': qb_stats['completions'].sum() / qb_stats['attempts'].sum()
                    if qb_stats['attempts'].sum() > 0 else 0.60
            }
        else:
            return {'passer_rating': 85.0, 'completion_pct': 0.60}

    def _get_referee_tendency(self, referee: str) -> float:
        """
        Get referee's historical scoring tendency

        Returns:
            Average points above/below vegas total for this ref
            Positive = games go over, Negative = games go under
        """
        # Calculate from historical games
        ref_games = self.schedules[
            (self.schedules['referee'] == referee) &
            (self.schedules['home_score'].notna()) &
            (self.schedules['total_line'].notna())
        ]

        if len(ref_games) > 0:
            ref_games = ref_games.copy()
            ref_games['actual_total'] = ref_games['home_score'] + ref_games['away_score']
            ref_games['total_error'] = ref_games['actual_total'] - ref_games['total_line']
            return ref_games['total_error'].mean()
        else:
            return 0.0  # No data, assume neutral

    def build_training_dataset(self,
                               start_season: int,
                               end_season: int,
                               min_week: int = 4) -> pd.DataFrame:
        """
        Build full training dataset for multiple seasons

        Args:
            start_season: First season to include
            end_season: Last season to include
            min_week: Minimum week (need history for features)

        Returns:
            DataFrame with features and targets
        """
        print(f"Building training dataset: {start_season}-{end_season}")

        games = self.schedules[
            (self.schedules['season'] >= start_season) &
            (self.schedules['season'] <= end_season) &
            (self.schedules['week'] >= min_week) &
            (self.schedules['game_type'] == 'REG') &
            (self.schedules['home_score'].notna())
        ].copy()

        print(f"Found {len(games)} games to process")

        all_features = []

        for idx, game in games.iterrows():
            # Build features for this game
            features = self.build_game_features(
                season=game['season'],
                week=game['week'],
                home_team=game['home_team'],
                away_team=game['away_team']
            )

            # Add identifiers
            features['game_id'] = game['game_id']
            features['season'] = game['season']
            features['week'] = game['week']
            features['home_team'] = game['home_team']
            features['away_team'] = game['away_team']

            # Add targets
            features['actual_margin'] = game['home_score'] - game['away_score']
            features['actual_total'] = game['home_score'] + game['away_score']
            features['spread_line'] = game['spread_line']
            features['total_line'] = game['total_line']

            all_features.append(features)

            if len(all_features) % 100 == 0:
                print(f"  Processed {len(all_features)} games...")

        df = pd.DataFrame(all_features)
        print(f"✓ Training dataset built: {len(df)} games, {len(df.columns)} features")

        return df


# Convenience function
def build_comprehensive_features(season: int, week: int,
                                 home_team: str, away_team: str,
                                 data_dir: str = '.') -> Dict:
    """
    Build comprehensive feature set for a single game

    Args:
        season: Season year
        week: Week number
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        data_dir: Path to data files

    Returns:
        Dictionary of features
    """
    builder = ComprehensiveFeatureBuilder(data_dir)
    builder.load_all_data()
    return builder.build_game_features(season, week, home_team, away_team)
