#!/usr/bin/env python
"""
PredictionTracker Benchmarking CLI for Ball Knower v1.2

Run benchmarks comparing Ball Knower predictions against PredictionTracker's
crowd of models using the canonical v1.2 dataset WITH actual game scores.

Usage:
    python src/run_predictiontracker_benchmarks.py \
        --pt_csv data/external/predictiontracker_nfl_2024.csv \
        --output_dir data/benchmarks \
        --outlier_threshold 4.0

This script:
    1. Loads the canonical Ball Knower v1.2 game-level frame (nflverse + nfelo)
    2. Loads a PredictionTracker CSV with model consensus predictions
    3. Merges on (season, week, home_team, away_team) for precise 1:1 matching
    4. Computes MAE vs actual game results for PT, Vegas, and BK (if available)
    5. Flags BK outlier games where BK differs significantly from PT consensus
    6. Writes merged data and summary metrics to CSV
    7. Prints a concise text report

Output files:
    - predictiontracker_merged_{season}.csv: full merged dataset
    - predictiontracker_summary_{season}.csv: summary metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
    print("BALL KNOWER v1.2 vs PREDICTIONTRACKER BENCHMARK")
    print("="*80)

    # Load canonical v1.2 games frame WITH BK predictions
    print(f"\n[1/5] Loading canonical Ball Knower v1.2 game-level frame...")
    games = v1_2.build_training_frame()
    print(f"  ✓ Loaded {len(games):,} games from {games['season'].min()}-{games['season'].max()}")

    # Load v1.2 model and add BK predictions
    print(f"\n[2/5] Loading Ball Knower v1.2 model and generating predictions...")
    try:
        coefficients, intercept = v1_2.load_v1_2_model()
        print(f"  ✓ Loaded model with {len(coefficients)} features")
        print(f"    Features: {', '.join(coefficients.keys())}")

        games = v1_2.add_bk_predictions(games, coefficients, intercept)

        if 'bk_line' not in games.columns:
            raise ValueError("Failed to add bk_line column to dataset")

        print(f"  ✓ Generated {games['bk_line'].notna().sum():,} BK predictions")
        print(f"    BK line range: [{games['bk_line'].min():.2f}, {games['bk_line'].max():.2f}]")
    except FileNotFoundError as e:
        print(f"\n  ERROR: Could not load v1.2 model")
        print(f"  {e}")
        print(f"\n  Please ensure output/ball_knower_v1_2_model.json exists.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: Failed to generate BK predictions")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load PredictionTracker
    print(f"\n[3/5] Loading PredictionTracker predictions from: {pt_csv_path}")
    pt_df = pt_bench.load_predictiontracker_csv(pt_csv_path)
    print(f"  ✓ Loaded {len(pt_df):,} PredictionTracker predictions")
    if 'pt_spread_std' in pt_df.columns:
        print(f"  ✓ Model disagreement (std) available")

    # Merge and compute metrics
    print(f"\n[4/5] Merging frames and computing benchmark metrics...")
    merged = pt_bench.merge_with_bk_games(
        pt_df,
        bk_games=games,
        outlier_threshold=outlier_threshold,
    )

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
    print(f"\n[5/5] Writing benchmark outputs...")
    merged_path = output_dir / f"predictiontracker_merged_{season_label}.csv"
    merged.to_csv(merged_path, index=False)
    print(f"  ✓ Wrote merged data: {merged_path}")
    print(f"    Rows: {len(merged):,}")
    print(f"    Columns: {len(merged.columns)}")

    # Compute summary metrics
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

    print("\n" + "-"*80)
    print("MEAN ABSOLUTE ERROR (MAE) METRICS")
    print("-"*80)

    # Print clean comparison table
    print("\nMetric                              Value")
    print("-" * 50)

    # MAE vs Actual
    if 'mae_bk_vs_actual' in summary:
        print(f"MAE: BK vs Actual              {summary['mae_bk_vs_actual']:>8.2f} points")
    else:
        print(f"MAE: BK vs Actual              {'N/A':>8}")

    if 'mae_pt_vs_actual' in summary:
        print(f"MAE: PT vs Actual              {summary['mae_pt_vs_actual']:>8.2f} points")
    else:
        print(f"MAE: PT vs Actual              {'N/A':>8}")

    if 'mae_vegas_vs_actual' in summary:
        print(f"MAE: Vegas vs Actual           {summary['mae_vegas_vs_actual']:>8.2f} points")
    else:
        print(f"MAE: Vegas vs Actual           {'N/A':>8}")

    print("")  # Blank line separator

    # MAE vs Vegas
    if 'mae_bk_vs_vegas' in summary:
        print(f"MAE: BK vs Vegas               {summary['mae_bk_vs_vegas']:>8.2f} points")
    else:
        print(f"MAE: BK vs Vegas               {'N/A':>8}")

    if 'mae_pt_vs_vegas' in summary:
        print(f"MAE: PT vs Vegas               {summary['mae_pt_vs_vegas']:>8.2f} points")
    else:
        print(f"MAE: PT vs Vegas               {'N/A':>8}")

    print("-" * 50)

    if 'bk_vs_pt_mean_diff' in summary and 'bk_vs_pt_mae_diff' in summary:
        print(f"\n--- BK vs PT Comparison ---")
        print(f"  Mean difference (BK - PT): {summary['bk_vs_pt_mean_diff']:+.2f} points")
        print(f"  Mean absolute difference:  {summary['bk_vs_pt_mae_diff']:.2f} points")

    if 'bk_outlier_count' in summary:
        print(f"\n--- BK Outlier Analysis (threshold: {outlier_threshold} pts) ---")
        print(f"  Outlier games: {summary['bk_outlier_count']:.0f} ({summary.get('bk_outlier_pct', 0):.1f}%)")

    # Show sample of matched games
    print(f"\n--- Sample of Matched Games ---")
    sample_cols = ['season', 'week', 'away_team', 'home_team', 'home_margin',
                   'vegas_closing_spread', 'pt_spread', 'bk_line',
                   'mae_pt_vs_actual', 'mae_vegas_vs_actual', 'mae_bk_vs_actual']
    available_cols = [c for c in sample_cols if c in merged.columns]

    if available_cols:
        sample = merged[available_cols].head(5).copy()
        # Round numeric columns
        for col in sample.columns:
            if sample[col].dtype in ['float64', 'float32']:
                sample[col] = sample[col].round(2)
        print(sample.to_string(index=False))

    # Show sample of unmatched PT games (if any)
    unmatched_pt = pt_df[~pt_df.set_index(['season', 'week', 'home_team', 'away_team']).index.isin(
        merged.set_index(['season', 'week', 'home_team', 'away_team']).index
    )]

    if len(unmatched_pt) > 0:
        print(f"\n--- Sample of Unmatched PT Games ({len(unmatched_pt)} total) ---")
        unmatched_cols = ['season', 'week', 'away_team', 'home_team', 'pt_spread']
        unmatched_available = [c for c in unmatched_cols if c in unmatched_pt.columns]
        if unmatched_available:
            unmatched_sample = unmatched_pt[unmatched_available].head(5).copy()
            for col in unmatched_sample.columns:
                if unmatched_sample[col].dtype in ['float64', 'float32']:
                    unmatched_sample[col] = unmatched_sample[col].round(2)
            print(unmatched_sample.to_string(index=False))

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
    - predictiontracker_merged_{season}.csv: full merged dataset with MAE metrics
    - predictiontracker_summary_{season}.csv: summary statistics
        """
    )

    parser.add_argument(
        "--pt_csv",
        required=True,
        help="Path to PredictionTracker NFL predictions CSV",
    )

    parser.add_argument(
        "--output_dir",
        default="data/benchmarks",
        help="Directory to write benchmark outputs (default: data/benchmarks)",
    )

    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=4.0,
        help="Absolute difference threshold (in points) for flagging BK outliers (default: 4.0)",
    )

    args = parser.parse_args()

    # Validate PT CSV exists
    if not Path(args.pt_csv).exists():
        print(f"ERROR: PredictionTracker CSV not found: {args.pt_csv}")
        sys.exit(1)

    try:
        run_benchmark(
            pt_csv_path=args.pt_csv,
            output_dir=args.output_dir,
            outlier_threshold=args.outlier_threshold,
        )
    except Exception as e:
        print(f"\nERROR: Benchmark failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
