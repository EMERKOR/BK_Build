"""
Get actual Vegas lines for Week 11 2025 from nflverse
"""

import nfl_data_py as nfl
import pandas as pd

print("Loading 2025 NFL schedule with Vegas lines...")

try:
    # Load 2025 schedule
    schedules = nfl.import_schedules([2025])

    # Filter for Week 11
    week_11 = schedules[schedules['week'] == 11].copy()

    print(f"\nFound {len(week_11)} games in Week 11")

    # Show columns available
    print(f"\nAvailable columns:")
    print(list(week_11.columns))

    # Show spread data
    print(f"\nWeek 11 Games with Spreads:")

    result = week_11[[
        'away_team', 'home_team', 'spread_line', 'total_line',
        'away_moneyline', 'home_moneyline', 'gameday'
    ]].copy()

    result = result.sort_values('gameday')

    print(result.to_string(index=False))

    # Save to CSV
    result.to_csv('data/current_season/vegas_lines_week_11.csv', index=False)
    print(f"\nSaved to data/current_season/vegas_lines_week_11.csv")

except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative approach...")

    # Try import_sc_lines if available
    try:
        lines = nfl.import_sc_lines([2025])
        print(f"\nScoring lines columns:")
        print(list(lines.columns))
        print(lines.head())
    except Exception as e2:
        print(f"Also failed: {e2}")
