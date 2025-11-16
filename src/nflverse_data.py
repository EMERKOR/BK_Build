"""
nflverse Data Access Module

Comprehensive access to all nflverse datasets.
Caches locally, auto-updates weekly.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class NFLVerseData:
    """
    Complete access to nflverse data repository.
    """

    BASE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data"

    DATASETS = {
        'games': f'{BASE_URL}/games.csv',
        'rosters': f'{BASE_URL}/rosters.csv',
        'standings': f'{BASE_URL}/standings.csv',
        'draft_picks': f'{BASE_URL}/draft_picks.csv',
        'draft_values': f'{BASE_URL}/draft_values.csv',
        'teams': f'{BASE_URL}/teams.csv',
    }

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / 'data' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _should_refresh(self, cache_file, max_age_hours=24):
        """Check if cached file should be refreshed."""
        if not cache_file.exists():
            return True

        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age > timedelta(hours=max_age_hours)

    def load(self, dataset_name, force_refresh=False):
        """
        Load any nflverse dataset.

        Args:
            dataset_name: Name from DATASETS dict
            force_refresh: Force download even if cached

        Returns:
            pd.DataFrame
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}")

        cache_file = self.cache_dir / f'{dataset_name}.csv'

        if not force_refresh and cache_file.exists() and not self._should_refresh(cache_file):
            return pd.read_csv(cache_file)

        # Download
        url = self.DATASETS[dataset_name]
        print(f"Downloading {dataset_name} from nflverse...")

        df = pd.read_csv(url)
        df.to_csv(cache_file, index=False)

        print(f"✓ Cached {len(df):,} rows to {cache_file}")
        return df

    # Convenience methods for common datasets

    def games(self, season=None, week=None):
        """Get games with optional filtering."""
        df = self.load('games')

        if season:
            df = df[df['season'] == season]
        if week:
            df = df[df['week'] == week]

        return df

    def rosters(self, season=None, team=None):
        """Get rosters with optional filtering."""
        df = self.load('rosters')

        if season:
            df = df[df['season'] == season]
        if team:
            df = df[df['team'] == team]

        return df

    def standings(self, season=None):
        """Get standings."""
        df = self.load('standings')

        if season:
            df = df[df['season'] == season]

        return df

    def draft_picks(self, season=None):
        """Get draft picks."""
        df = self.load('draft_picks')

        if season:
            df = df[df['season'] == season]

        return df

    def teams(self):
        """Get team metadata."""
        return self.load('teams')

    # Analysis helpers

    def get_team_form(self, team, season, through_week):
        """
        Get recent form for a team.

        Args:
            team: Team abbreviation
            season: Season year
            through_week: Last week to include

        Returns:
            dict with recent stats
        """
        games = self.games(season=season)
        games = games[games['week'] <= through_week]

        # Get games for this team
        team_games = games[
            ((games['home_team'] == team) | (games['away_team'] == team))
        ].copy()

        team_games['is_home'] = team_games['home_team'] == team
        team_games['team_score'] = team_games.apply(
            lambda r: r['home_score'] if r['is_home'] else r['away_score'], axis=1
        )
        team_games['opp_score'] = team_games.apply(
            lambda r: r['away_score'] if r['is_home'] else r['home_score'], axis=1
        )

        team_games['won'] = team_games['team_score'] > team_games['opp_score']
        team_games['margin'] = team_games['team_score'] - team_games['opp_score']

        # Calculate form stats
        last_5 = team_games.tail(5)

        return {
            'games_played': len(team_games),
            'wins': team_games['won'].sum(),
            'losses': (~team_games['won']).sum(),
            'ppg': team_games['team_score'].mean(),
            'ppg_allowed': team_games['opp_score'].mean(),
            'margin': team_games['margin'].mean(),
            'last_5_wins': last_5['won'].sum(),
            'last_5_margin': last_5['margin'].mean(),
        }

    def get_matchup_history(self, team1, team2, n_games=10):
        """
        Get head-to-head history between two teams.

        Args:
            team1, team2: Team abbreviations
            n_games: Number of recent games

        Returns:
            pd.DataFrame with matchup history
        """
        games = self.load('games')

        matchups = games[
            ((games['home_team'] == team1) & (games['away_team'] == team2)) |
            ((games['home_team'] == team2) & (games['away_team'] == team1))
        ].copy()

        matchups = matchups.sort_values('gameday', ascending=False).head(n_games)

        return matchups[['season', 'week', 'gameday', 'away_team', 'home_team',
                        'away_score', 'home_score', 'spread_line']]


# Convenience instance
nflverse = NFLVerseData()


if __name__ == "__main__":
    print("Testing nflverse data access...\n")

    # Test basic loading
    games = nflverse.games(season=2025, week=11)
    print(f"Week 11 2025: {len(games)} games")
    print(games[['away_team', 'home_team', 'spread_line']].to_string(index=False))

    # Test team form
    print("\nKC form through Week 10:")
    form = nflverse.get_team_form('KC', 2025, 10)
    for k, v in form.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")

    # Test matchup history
    print("\nKC vs BUF history (last 5):")
    history = nflverse.get_matchup_history('KC', 'BUF', 5)
    print(history.to_string(index=False))
