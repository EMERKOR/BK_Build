"""
Setup Team Profiles Script

Creates team profile directory structure and skeleton profile.md files
for all teams in the Ball Knower canonical games dataset.

Usage:
    python src/setup_team_profiles.py --season 2025
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Set

# Use the canonical team list from team_mapping module
from team_mapping import NFL_DATA_PY_TEAMS


def get_teams_from_nfelo(season: int) -> Set[str]:
    """
    Load canonical games dataset from nfelo and extract unique team codes.

    Args:
        season: NFL season year to filter for

    Returns:
        Set of team codes (e.g., {'LAR', 'BUF', 'MIN', ...})
    """
    # Load nfelo historical data (same source as ball_knower.datasets.v1_2)
    data_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
    df = pd.read_csv(data_url)

    # Extract season/week/teams from game_id (format: {season}_{week}_{away_team}_{home_team})
    df[['season', 'week', 'away_team', 'home_team']] = \
        df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
    df['season'] = df['season'].astype(int)

    # Filter to requested season
    df = df[df['season'] == season].copy()

    # Get unique teams
    teams = set(df['home_team'].unique()) | set(df['away_team'].unique())

    # Remove any NaN values
    teams = {t for t in teams if pd.notna(t)}

    return teams


def create_profile_md(team_code: str, output_dir: Path) -> None:
    """
    Create a skeleton profile.md file for a team.

    Args:
        team_code: Team code (e.g., 'LAR', 'BUF')
        output_dir: Directory to write profile.md to
    """
    profile_path = output_dir / "profile.md"

    # Skip if profile.md already exists (idempotent)
    if profile_path.exists():
        print(f"  ✓ Profile already exists for {team_code}")
        return

    content = f"""# {team_code} – Ball Knower Team Profile

**Team code:** {team_code}
**Last major update:** TBD

---
## 1. Identity Snapshot

_(This will be populated later by LLM-generated summaries and analyst research.)_

---
## 2. Offensive Overview

_(To be auto-filled from research + FantasyPoints-assisted analysis.)_

---
## 3. Defensive Overview

_(To be auto-filled from research + FantasyPoints-assisted analysis.)_

---
## 4. Coaching & Philosophy

_(To be auto-filled.)_

---
## 5. Season Arc & Injuries (Macro)

_(To be auto-filled.)_

---
## 6. Motivation & Stability Notes

_(To be auto-filled.)_

---
## 7. Notes for Ball Knower Modeling

_(Bullet points that the LLM layer will maintain.)_
"""

    profile_path.write_text(content)
    print(f"  ✓ Created profile.md for {team_code}")


def setup_team_profile_dirs(season: int, use_canonical_teams: bool = False) -> None:
    """
    Create team profile directory structure for all teams.

    Args:
        season: NFL season year
        use_canonical_teams: If True, use NFL_DATA_PY_TEAMS instead of nfelo data
    """
    # Get project root
    project_root = Path(__file__).resolve().parents[1]
    subjective_dir = project_root / "subjective" / "team_profiles"

    # Ensure subjective/team_profiles exists
    subjective_dir.mkdir(parents=True, exist_ok=True)

    # Get team list
    if use_canonical_teams:
        print(f"Using canonical NFL team list ({len(NFL_DATA_PY_TEAMS)} teams)")
        teams = set(NFL_DATA_PY_TEAMS)
    else:
        print(f"Loading teams from nfelo data for {season} season...")
        teams = get_teams_from_nfelo(season)
        print(f"Found {len(teams)} teams in {season} season")

    # Create directory and profile.md for each team
    for team_code in sorted(teams):
        team_dir = subjective_dir / team_code
        weeks_dir = team_dir / "weeks"

        # Create directories
        weeks_dir.mkdir(parents=True, exist_ok=True)

        # Create profile.md if it doesn't exist
        create_profile_md(team_code, team_dir)

    print(f"\n✅ Team profile setup complete!")
    print(f"📂 Created {len(teams)} team directories in: {subjective_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup team profile directories and skeleton profile.md files"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="NFL season year (default: 2025)"
    )
    parser.add_argument(
        "--use-canonical",
        action="store_true",
        help="Use canonical NFL team list instead of loading from nfelo data"
    )

    args = parser.parse_args()

    setup_team_profile_dirs(args.season, args.use_canonical)
