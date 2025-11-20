#!/usr/bin/env python
"""
Apply structural edge factors to Week 12 team power ratings.

This script:
1. Loads the base team power ratings
2. Loads the structural factors
3. Merges them and updates structural columns
4. Recomputes subjective_total and bk_blended_rating
5. Overwrites the power ratings file
"""

import pandas as pd
from pathlib import Path

# File paths
POWER_RATINGS_PATH = Path("subjective/team_power_ratings_2025_week12.csv")
STRUCTURAL_PATH = Path("subjective/structural_factors_2025_week12.csv")

# Weights for blended rating
MARKET_WEIGHT = 0.40
OBJECTIVE_WEIGHT = 0.30
ANALYST_WEIGHT = 0.20
SUBJECTIVE_WEIGHT = 0.10


def main():
    print("="*70)
    print("Applying Structural Edge to Week 12 Team Power Ratings")
    print("="*70)
    print()

    # Load power ratings
    print(f"Loading power ratings from: {POWER_RATINGS_PATH}")
    df_ratings = pd.read_csv(POWER_RATINGS_PATH)
    print(f"  Loaded {len(df_ratings)} teams")
    print()

    # Load structural factors
    print(f"Loading structural factors from: {STRUCTURAL_PATH}")
    df_structural = pd.read_csv(STRUCTURAL_PATH)
    print(f"  Loaded {len(df_structural)} teams")
    print()

    # Merge on season, week, team_code
    print("Merging datasets...")
    merge_keys = ["season", "week", "team_code"]

    # Drop structural columns from ratings if they exist
    structural_cols = [
        "offense_series_score",
        "defense_series_score",
        "ol_structure_score",
        "coaching_edge_score",
        "structural_edge",
        "notes"
    ]

    cols_to_drop = [col for col in structural_cols if col in df_ratings.columns]
    if cols_to_drop:
        df_ratings = df_ratings.drop(columns=cols_to_drop)

    # Merge
    df = df_ratings.merge(df_structural, on=merge_keys, how="left")
    print(f"  Merged {len(df)} teams")
    print()

    # Recompute subjective_total (now includes structural_edge)
    print("Recomputing subjective_total...")
    df["subjective_total"] = (
        df["subjective_health_adj"]
        + df["subjective_form_adj"]
        + df["structural_edge"]
    )
    print(f"  Subjective total range: [{df['subjective_total'].min():.2f}, {df['subjective_total'].max():.2f}]")
    print()

    # Recompute bk_blended_rating
    print("Recomputing bk_blended_rating...")
    df["bk_blended_rating"] = (
        MARKET_WEIGHT * df["market_rating"]
        + OBJECTIVE_WEIGHT * df["objective_composite"]
        + ANALYST_WEIGHT * df["analyst_composite_rating"]
        + SUBJECTIVE_WEIGHT * df["subjective_total"]
    )
    print(f"  Blended rating range: [{df['bk_blended_rating'].min():.2f}, {df['bk_blended_rating'].max():.2f}]")
    print()

    # Sanity checks
    print("Running sanity checks...")
    assert len(df) == 32, f"Expected 32 teams, got {len(df)}"
    print(f"  ✓ Have 32 teams")

    assert df["bk_blended_rating"].notna().all(), "Found NaN values in bk_blended_rating"
    print(f"  ✓ No NaN values in bk_blended_rating")
    print()

    # Reorder columns to match original structure
    cols_order = [
        'season', 'week', 'team_code', 'team_name',
        'market_rating',
        'nfelo_value', 'substack_points', 'objective_composite',
        'athletic_rank', 'pff_rank', 'analyst_composite_rating',
        'offense_series_score', 'defense_series_score',
        'ol_structure_score', 'coaching_edge_score', 'structural_edge',
        'subjective_health_adj', 'subjective_form_adj', 'subjective_total',
        'bk_blended_rating',
        'notes'
    ]

    df = df[cols_order]

    # Sort by blended rating
    df = df.sort_values("bk_blended_rating", ascending=False)

    # Save back to file
    print(f"Saving updated ratings to: {POWER_RATINGS_PATH}")
    df.to_csv(POWER_RATINGS_PATH, index=False)
    print("  ✓ File saved")
    print()

    # Print top 10
    print("="*70)
    print("TOP 10 TEAMS BY BK BLENDED RATING (with structural edge)")
    print("="*70)
    print()

    top10 = df.head(10)[[
        'team_code', 'team_name',
        'market_rating', 'objective_composite', 'analyst_composite_rating',
        'structural_edge', 'subjective_total',
        'bk_blended_rating'
    ]]

    print(top10.to_string(index=False))
    print()
    print("="*70)


if __name__ == "__main__":
    main()
