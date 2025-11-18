#!/usr/bin/env python
"""
PredictionTracker Benchmarking CLI

Run benchmarks comparing Ball Knower predictions against PredictionTracker's
crowd of models.

Usage:
    python src/run_predictiontracker_benchmarks.py \
        --pt_csv data/external/predictiontracker_nfl_2024.csv \
        --output_dir data/benchmarks \
        --outlier_threshold 4.0

This script:
    1. Loads the canonical Ball Knower v1.2 game-level training frame
    2. Loads a PredictionTracker CSV with model consensus predictions
    3. Merges and computes benchmark metrics (MAE for PT, BK, Vegas)
    4. Flags BK outlier games where BK differs significantly from PT consensus
    5. Writes merged data and summary metrics to CSV
    6. Prints a concise text report

Output files:
    - predictiontracker_merged_{season}.csv: full merged dataset
    - predictiontracker_summary_{season}.csv: summary metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ball_knower.benchmarks import predictiontracker as pt_bench
from ball_knower.datasets import v1_2


def run_benchmark(
    pt_csv_path: str,
    output_dir: str = "data/benchmarks",
    outlier_threshold: float = 4.0,
) -> None:
    """
    Run PredictionTracker benchmark against Ball Knower v1.2.

    Parameters
    ----------
    pt_csv_path : str
        Path to PredictionTracker NFL predictions CSV
    output_dir : str
        Directory to write benchmark outputs
    outlier_threshold : float
        Absolute difference threshold for BK vs PT to flag outliers
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("BALL KNOWER vs PREDICTIONTRACKER BENCHMARK")
    print("="*80)

    # Load canonical v1.2 games frame
    print(f"\n[1/4] Loading canonical Ball Knower v1.2 game-level frame...")
    games = v1_2.build_training_frame()
    print(f"  ✓ Loaded {len(games):,} games from {games['season'].min()}-{games['season'].max()}")

    # Load PredictionTracker
    print(f"\n[2/4] Loading PredictionTracker predictions from: {pt_csv_path}")
    pt_df = pt_bench.load_predictiontracker_csv(pt_csv_path)
    print(f"  ✓ Loaded {len(pt_df):,} PredictionTracker predictions")
    if 'pt_pred_std' in pt_df.columns:
        print(f"  ✓ Model disagreement (std) available")

    # Merge and compute metrics
    print(f"\n[3/4] Merging frames and computing benchmark metrics...")
    merged = pt_bench.merge_with_bk_games(
        pt_df,
        bk_games=games,
        outlier_threshold=outlier_threshold,
    )
    print(f"  ✓ Merged {len(merged):,} games")

    # Count how many have PT predictions
    n_with_pt = merged['pt_pred_avg'].notna().sum()
    print(f"  ✓ {n_with_pt:,} games ({n_with_pt/len(merged)*100:.1f}%) have PT predictions")

    # Try to infer season range for naming
    if 'season' in merged.columns:
        seasons = sorted(merged['season'].dropna().unique().tolist())
        if seasons:
            season_label = f"{min(seasons)}-{max(seasons)}" if len(seasons) > 1 else str(seasons[0])
        else:
            season_label = "unknown"
    else:
        season_label = "unknown"

    # Write merged data
    merged_path = output_dir / f"predictiontracker_merged_{season_label}.csv"
    merged.to_csv(merged_path, index=False)
    print(f"  ✓ Wrote merged data: {merged_path}")

    # Compute summary metrics
    print(f"\n[4/4] Computing summary metrics...")
    summary_df = pt_bench.compute_summary_metrics(merged)
    summary_path = output_dir / f"predictiontracker_summary_{season_label}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Wrote summary: {summary_path}")

    # Print concise text report
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    summary = summary_df.iloc[0].to_dict()

    print(f"\nGames analyzed: {summary.get('n_games', 0):,.0f}")
    print(f"Games with PT predictions: {summary.get('n_games_with_pt', 0):,.0f}")

    print("\n--- Mean Absolute Error vs Actual Margin ---")

    if 'pt_mae_vs_margin' in summary:
        print(f"  PredictionTracker: {summary['pt_mae_vs_margin']:.2f} points")

    if 'bk_mae_vs_margin' in summary:
        print(f"  Ball Knower v1.2:  {summary['bk_mae_vs_margin']:.2f} points")
    else:
        print(f"  Ball Knower v1.2:  N/A (no BK predictions in frame)")

    if 'vegas_mae_vs_margin' in summary:
        print(f"  Vegas closing:     {summary['vegas_mae_vs_margin']:.2f} points")

    if 'bk_vs_pt_mean_diff' in summary and 'bk_vs_pt_mae_diff' in summary:
        print(f"\n--- BK vs PT Comparison ---")
        print(f"  Mean difference (BK - PT): {summary['bk_vs_pt_mean_diff']:+.2f} points")
        print(f"  Mean absolute difference:  {summary['bk_vs_pt_mae_diff']:.2f} points")

    if 'bk_outlier_count' in summary:
        print(f"\n--- BK Outlier Analysis (threshold: {outlier_threshold} pts) ---")
        print(f"  Outlier games: {summary['bk_outlier_count']:.0f} ({summary.get('bk_outlier_pct', 0):.1f}%)")

    # Show top outliers if available
    if 'bk_outlier_flag' in merged.columns and 'bk_vs_pt_diff' in merged.columns:
        outliers = merged[merged['bk_outlier_flag'] == True].copy()
        if len(outliers) > 0:
            print(f"\nTop 10 BK outliers (largest |BK - PT| differences):")
            top_outliers = outliers.nlargest(10, 'bk_vs_pt_diff', keep='all')[
                ['game_id', 'home_team', 'away_team', 'bk_line', 'pt_pred_avg', 'bk_vs_pt_diff']
            ].copy() if 'game_id' in outliers.columns else outliers.nlargest(10, 'bk_vs_pt_diff', keep='all')[
                ['season', 'week', 'home_team', 'away_team', 'bk_line', 'pt_pred_avg', 'bk_vs_pt_diff']
            ].copy()

            # Round numeric columns for display
            for col in ['bk_line', 'pt_pred_avg', 'bk_vs_pt_diff']:
                if col in top_outliers.columns:
                    top_outliers[col] = top_outliers[col].round(2)

            print(top_outliers.to_string(index=False, max_rows=10))

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nMerged data: {merged_path}")
    print(f"Summary:     {summary_path}\n")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run PredictionTracker benchmarks against Ball Knower v1.2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic benchmark
  python src/run_predictiontracker_benchmarks.py \\
      --pt_csv data/external/predictiontracker_nfl_2024.csv

  # Custom output directory and outlier threshold
  python src/run_predictiontracker_benchmarks.py \\
      --pt_csv data/external/pt_2024.csv \\
      --output_dir output/benchmarks \\
      --outlier_threshold 5.0

Output:
  Creates two CSV files in the output directory:
    - predictiontracker_merged_{season}.csv: full merged dataset
    - predictiontracker_summary_{season}.csv: summary metrics
        """
    )

    parser.add_argument(
        "--pt_csv",
        required=True,
        help="Path to PredictionTracker NFL predictions CSV (manually downloaded).",
    )

    parser.add_argument(
        "--output_dir",
        default="data/benchmarks",
        help="Directory to write benchmark outputs (default: data/benchmarks).",
    )

    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=4.0,
        help="Absolute difference threshold (in points) for BK vs PT to flag outliers (default: 4.0).",
    )

    args = parser.parse_args()

    try:
        run_benchmark(
            pt_csv_path=args.pt_csv,
            output_dir=args.output_dir,
            outlier_threshold=args.outlier_threshold,
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    main()
