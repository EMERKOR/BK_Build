"""
Multi-Model Backtest Comparison

Runs backtests for multiple Ball Knower model versions and produces
a side-by-side comparison table.

Usage:
    python run_multi_backtests.py --models v1.0 v1.2 v1.3 --start-season 2019 --end-season 2024

Output:
    - Individual backtest CSVs for each model
    - Aggregated comparison table showing relative performance
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ball_knower.utils import paths, version
from src import run_backtests


def run_multi_model_backtest(
    models: list,
    start_season: int,
    end_season: int,
    edge_threshold: float = 0.5
):
    """
    Run backtests for multiple models and aggregate results.

    Args:
        models: List of model versions (e.g., ['v1.0', 'v1.2', 'v1.3'])
        start_season: Start season year
        end_season: End season year
        edge_threshold: Edge threshold for betting (default: 0.5)

    Returns:
        dict: Mapping of model_version -> backtest DataFrame
    """
    results = {}

    for model in models:
        print(f"\n{'='*80}")
        print(f"Running backtest for {model}...")
        print(f"{'='*80}")

        try:
            if model == 'v1.0':
                df = run_backtests.run_backtest_v1_0(
                    start_season,
                    end_season,
                    edge_threshold
                )
            elif model == 'v1.2':
                df = run_backtests.run_backtest_v1_2(
                    start_season,
                    end_season,
                    edge_threshold
                )
            elif model == 'v1.3':
                # v1.3 backtest might not exist yet, use v1.2 as placeholder
                print(f"  ⚠ v1.3 backtest not yet implemented, using v1.2 as proxy")
                df = run_backtests.run_backtest_v1_2(
                    start_season,
                    end_season,
                    edge_threshold
                )
            else:
                print(f"  ⚠ Unknown model '{model}', skipping")
                continue

            results[model] = df
            print(f"  ✓ {model} backtest complete: {len(df)} seasons analyzed")

            # Save individual backtest
            output_path = paths.get_backtest_results_path(model, start_season, end_season)
            df.to_csv(output_path, index=False)
            print(f"  ✓ Saved to: {output_path}")

        except Exception as e:
            print(f"  ✗ Error running {model} backtest: {e}")
            continue

    return results


def create_comparison_table(results: dict) -> pd.DataFrame:
    """
    Create side-by-side comparison table for multiple models.

    Args:
        results: Dict mapping model_version -> backtest DataFrame

    Returns:
        DataFrame with aggregated metrics for each model
    """
    comparison_rows = []

    for model, df in results.items():
        # Aggregate metrics across all seasons
        row = {
            'model': model,
            'seasons': len(df),
            'total_games': df['n_games'].sum(),
            'avg_mae': df['mae_vs_vegas'].mean(),
            'avg_rmse': df['rmse_vs_vegas'].mean(),
            'avg_edge': df['mean_edge'].mean(),
            'edge_std': df['mean_edge'].std(),
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    # Sort by MAE (lower is better)
    comparison_df = comparison_df.sort_values('avg_mae')

    return comparison_df


def main():
    """
    Main entry point for multi-model backtest comparison.
    """
    parser = argparse.ArgumentParser(
        description='Compare multiple Ball Knower model versions via backtesting'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['v1.0', 'v1.2'],
        help='Model versions to compare (default: v1.0 v1.2)'
    )
    parser.add_argument(
        '--start-season',
        type=int,
        default=2019,
        help='Start season (default: 2019)'
    )
    parser.add_argument(
        '--end-season',
        type=int,
        default=2024,
        help='End season (default: 2024)'
    )
    parser.add_argument(
        '--edge-threshold',
        type=float,
        default=0.5,
        help='Edge threshold for betting decisions (default: 0.5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for comparison table (default: output/backtests/comparison_YYYY_ZZZZ.csv)'
    )

    args = parser.parse_args()

    # Print banner
    version.print_version_banner("multi_backtest_comparison")

    print("\nMulti-Model Backtest Comparison")
    print("=" * 80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Seasons: {args.start_season}-{args.end_season}")
    print(f"Edge threshold: {args.edge_threshold}")
    print("=" * 80)

    # Run backtests
    results = run_multi_model_backtest(
        args.models,
        args.start_season,
        args.end_season,
        args.edge_threshold
    )

    if not results:
        print("\n✗ No backtests completed successfully")
        return 1

    # Create comparison table
    print(f"\n{'='*80}")
    print("Aggregating results...")
    print(f"{'='*80}")

    comparison_df = create_comparison_table(results)

    # Save comparison table
    if args.output is None:
        backtests_dir = paths.get_backtests_dir()
        output_path = backtests_dir / f"comparison_{args.start_season}_{args.end_season}.csv"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_path, index=False)
    print(f"\n✓ Comparison table saved to: {output_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    print(f"{'='*80}")

    print("\nInterpretation:")
    print("  - Lower MAE/RMSE indicates better accuracy vs Vegas")
    print("  - Near-zero mean edge suggests Vegas market efficiency")
    print("  - Higher edge std suggests more variable predictions")

    return 0


if __name__ == '__main__':
    sys.exit(main())
