"""
Script to create synthetic games.csv fixture for testing

This creates a minimal CSV file with fake game data to allow
dataset builder tests to run without requiring real nfelo data.
"""

import pandas as pd
from pathlib import Path

# Create synthetic game data with real NFL team abbreviations
games_data = {
    'game_id': [
        '2023_01_BUF_KC',
        '2023_01_PHI_BUF',
        '2023_02_KC_PHI',
        '2023_02_BUF_KC',
        '2023_03_PHI_KC',
        '2023_03_BUF_PHI',
        '2024_01_KC_BUF',
        '2024_01_PHI_KC',
    ],
    'season': [2023, 2023, 2023, 2023, 2023, 2023, 2024, 2024],
    'week': [1, 1, 2, 2, 3, 3, 1, 1],
    'home_team': ['KC', 'BUF', 'PHI', 'KC', 'KC', 'PHI', 'BUF', 'KC'],
    'away_team': ['BUF', 'PHI', 'KC', 'BUF', 'PHI', 'BUF', 'KC', 'PHI'],
    'home_score': [24, 27, 21, 28, 17, 31, 23, 20],
    'away_score': [21, 24, 17, 24, 14, 28, 20, 17],
    'home_line_close': [-3.5, -4.0, -2.5, -5.0, -1.5, -6.0, -3.0, -2.0],
    'starting_nfelo_home': [1600, 1620, 1590, 1605, 1595, 1625, 1610, 1598],
    'starting_nfelo_away': [1550, 1560, 1570, 1555, 1565, 1550, 1585, 1575],
    # Additional columns for v1.2 dataset
    'home_bye_mod': [0, 0, 0, 0, 25, 0, 0, 0],
    'away_bye_mod': [0, 0, 0, -25, 0, 0, 0, 0],
    'div_game_mod': [0, 10, 0, 0, 10, 0, 0, 10],
    'dif_surface_mod': [0, 5, 0, -5, 0, 0, 5, 0],
    'home_time_advantage_mod': [0, 0, 15, 0, 0, 0, 0, 15],
    'home_538_qb_adj': [25, 30, 20, 28, 22, 32, 27, 24],
    'away_538_qb_adj': [20, 22, 18, 24, 19, 28, 25, 21],
}

# Create DataFrame
df = pd.DataFrame(games_data)

# Save to CSV
output_path = Path(__file__).parent / "nfelo_games.csv"
df.to_csv(output_path, index=False)

print(f"Created {output_path}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
