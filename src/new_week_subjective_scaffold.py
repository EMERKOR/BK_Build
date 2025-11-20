"""
New Week Subjective Scaffold Script

Creates weekly team update markdown files (week_XX.md) for all teams
involved in games during a specific season and week.

Usage:
    python src/new_week_subjective_scaffold.py --season 2025 --week 12
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set


def get_games_for_week(season: int, week: int) -> pd.DataFrame:
    """
    Load games from nfelo dataset for a specific season and week.

    Args:
        season: NFL season year
        week: NFL week number

    Returns:
        DataFrame with columns: game_id, season, week, away_team, home_team
    """
    # Load nfelo historical data (same source as ball_knower.datasets.v1_2)
    data_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
    df = pd.read_csv(data_url)

    # Extract season/week/teams from game_id (format: {season}_{week}_{away_team}_{home_team})
    df[['season', 'week', 'away_team', 'home_team']] = \
        df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # Filter to requested season and week
    df = df[(df['season'] == season) & (df['week'] == week)].copy()

    return df[['game_id', 'season', 'week', 'away_team', 'home_team']]


def get_opponent_for_team(games: pd.DataFrame, team: str) -> str:
    """
    Find the opponent for a team in the games DataFrame.

    Args:
        games: DataFrame of games for this week
        team: Team code

    Returns:
        Opponent team code, or "TBD" if not found
    """
    # Check if team is home
    home_game = games[games['home_team'] == team]
    if len(home_game) > 0:
        return home_game.iloc[0]['away_team']

    # Check if team is away
    away_game = games[games['away_team'] == team]
    if len(away_game) > 0:
        return away_game.iloc[0]['home_team']

    return "TBD"


def create_week_md(
    team_code: str,
    season: int,
    week: int,
    opponent: str,
    output_dir: Path
) -> None:
    """
    Create a skeleton week_XX.md file for a team.

    Args:
        team_code: Team code (e.g., 'LAR', 'BUF')
        season: NFL season year
        week: NFL week number
        opponent: Opponent team code
        output_dir: Directory to write week_XX.md to
    """
    # Format week with zero-padding (e.g., 01, 02, 12, 18)
    week_str = f"{week:02d}"
    week_path = output_dir / f"week_{week_str}.md"

    # Skip if week file already exists (idempotent)
    if week_path.exists():
        print(f"  ✓ Week {week} file already exists for {team_code}")
        return

    content = f"""# {team_code} – Week {week} Update ({season})

**Week:** {week}
**Season:** {season}
**Opponent:** {opponent}
**Game context:** TBD (Home/Away, spread, total)
**Current record before this game:** TBD
**Date of this update (real-world):** TBD

---
## 1. What Changed Since Last Week?

_(To be populated from voice transcripts / analyst notes.)_

---
## 2. Offense This Week

### 2.1 QB & Passing

### 2.2 Offensive Line

### 2.3 Pass Catchers

### 2.4 Run Game

---
## 3. Defense This Week

### 3.1 Front Seven

### 3.2 Secondary

---
## 4. Coaching, Scheme, and Game Management

---
## 5. Motivation, Stability, and Narrative

---
## 6. Slider Adjustments for This Week (Narrative Notes)

_(This is where the LLM layer will describe any changes to sliders in plain language.)_

---
## 7. Matchup-Relevant Notes

_(Anything especially relevant for this particular opponent or matchup.)_
"""

    week_path.write_text(content)
    print(f"  ✓ Created week_{week_str}.md for {team_code}")


def scaffold_weekly_team_markdowns(season: int, week: int) -> None:
    """
    Create weekly team update markdown files for all teams playing this week.

    Args:
        season: NFL season year
        week: NFL week number
    """
    # Get project root
    project_root = Path(__file__).resolve().parents[1]
    team_profiles_dir = project_root / "subjective" / "team_profiles"

    # Ensure team_profiles directory exists
    if not team_profiles_dir.exists():
        raise FileNotFoundError(
            f"Team profiles directory not found: {team_profiles_dir}\n"
            f"Please run 'python src/setup_team_profiles.py' first"
        )

    # Load games for this week
    print(f"Loading games for {season} season, week {week}...")
    games = get_games_for_week(season, week)

    if len(games) == 0:
        print(f"⚠️  No games found for {season} season, week {week}")
        return

    print(f"Found {len(games)} game(s) for week {week}")

    # Get all teams involved this week
    teams = set(games['home_team'].unique()) | set(games['away_team'].unique())
    teams = {t for t in teams if pd.notna(t)}  # Remove NaN values

    print(f"Creating weekly updates for {len(teams)} teams...\n")

    # Create week_XX.md for each team
    created_count = 0
    for team_code in sorted(teams):
        team_dir = team_profiles_dir / team_code
        weeks_dir = team_dir / "weeks"

        # Ensure weeks directory exists
        weeks_dir.mkdir(parents=True, exist_ok=True)

        # Find opponent
        opponent = get_opponent_for_team(games, team_code)

        # Create week markdown
        create_week_md(team_code, season, week, opponent, weeks_dir)
        created_count += 1

    print(f"\n✅ Weekly scaffold complete!")
    print(f"📂 Processed {created_count} team(s) for {season} season, week {week}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create weekly team update markdown files for a specific week"
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season year (e.g., 2025)"
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="NFL week number (e.g., 12)"
    )

    args = parser.parse_args()

    scaffold_weekly_team_markdowns(args.season, args.week)
