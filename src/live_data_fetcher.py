"""
Live Data Fetcher for Ball Knower

Automatically fetches current NFL data from nflverse and other sources.
No manual data entry required.
"""

import pandas as pd
import requests
from datetime import datetime
from pathlib import Path

class NFLDataFetcher:
    """
    Fetches live NFL data from nflverse GitHub repositories.
    """

    # nflverse raw data URLs
    GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    ROSTERS_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/rosters.csv"

    def __init__(self, cache_dir=None):
        """
        Args:
            cache_dir: Directory to cache downloaded files
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / 'data' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_schedules(self, season, force_refresh=False):
        """
        Fetch NFL schedules with Vegas lines for a given season.

        Args:
            season (int): NFL season year
            force_refresh (bool): Force download even if cached

        Returns:
            pd.DataFrame: Games with columns including spread_line, total_line, etc.
        """
        cache_file = self.cache_dir / f'schedules_{season}.csv'

        # Use cache if available and not forcing refresh
        if cache_file.exists() and not force_refresh:
            print(f"Loading schedules from cache: {cache_file}")
            return pd.read_csv(cache_file)

        # Fetch from nflverse
        print(f"Fetching {season} schedules from nflverse...")

        try:
            df = pd.read_csv(self.GAMES_URL)

            # Filter for requested season
            df = df[df['season'] == season].copy()

            # Ensure required columns exist
            required_cols = ['season', 'week', 'gameday', 'away_team', 'home_team']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Cache the result
            df.to_csv(cache_file, index=False)
            print(f"Cached {len(df)} games to {cache_file}")

            return df

        except Exception as e:
            raise Exception(f"Failed to fetch schedules: {e}")

    def get_current_week(self, season):
        """
        Determine current NFL week based on today's date.

        Args:
            season (int): NFL season year

        Returns:
            int: Current week number
        """
        schedules = self.fetch_schedules(season)
        schedules['gameday'] = pd.to_datetime(schedules['gameday'])

        today = datetime.now()

        # Find the week containing games closest to today
        future_games = schedules[schedules['gameday'] >= today]

        if len(future_games) > 0:
            return future_games.iloc[0]['week']

        # If no future games, return last week
        return schedules['week'].max()

    def get_week_games(self, season, week):
        """
        Get all games for a specific week with Vegas lines.

        Args:
            season (int): NFL season year
            week (int): Week number

        Returns:
            pd.DataFrame: Games for the week with spread_line, total_line, etc.
        """
        schedules = self.fetch_schedules(season)

        week_games = schedules[schedules['week'] == week].copy()

        print(f"Found {len(week_games)} games in {season} Week {week}")

        # Check if Vegas lines are available
        if 'spread_line' in week_games.columns:
            lines_available = week_games['spread_line'].notna().sum()
            print(f"Vegas lines available: {lines_available}/{len(week_games)} games")
        else:
            print("Warning: No spread_line column in data")

        return week_games

    def get_latest_lines(self, season, week):
        """
        Get the most recent Vegas lines for a week.

        Args:
            season (int): NFL season year
            week (int): Week number

        Returns:
            pd.DataFrame: Games with latest Vegas lines
        """
        games = self.get_week_games(season, week)

        # Select relevant columns
        result = games[['away_team', 'home_team', 'spread_line', 'total_line',
                       'away_moneyline', 'home_moneyline', 'gameday']].copy()

        result = result.sort_values('gameday')

        return result


def fetch_current_week_lines(season=2025):
    """
    Convenience function to get current week lines.

    Args:
        season (int): NFL season year

    Returns:
        pd.DataFrame: Current week games with Vegas lines
    """
    fetcher = NFLDataFetcher()
    current_week = fetcher.get_current_week(season)

    print(f"\nCurrent Week: {current_week}")

    lines = fetcher.get_latest_lines(season, current_week)

    return lines


if __name__ == "__main__":
    # Test the fetcher
    print("Testing NFL Data Fetcher...\n")

    try:
        lines = fetch_current_week_lines(2025)
        print("\nCurrent Week Lines:")
        print(lines.to_string(index=False))

    except Exception as e:
        print(f"Error: {e}")
