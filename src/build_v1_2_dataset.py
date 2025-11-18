"""
Build v1.2 Training Dataset

Generates the canonical v1.2 training dataset and saves it as Parquet.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))

    from ball_knower.datasets.v1_2 import save_training_frame
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "v1_2_training_dataset.parquet"

    print("\n" + "="*80)
    print("BALL KNOWER v1.2 - DATASET BUILDER")
    print("="*80)

    path = save_training_frame(
        output_path=OUTPUT_PATH,
        data_dir=DATA_DIR,
        start_season=2009,
        end_season=2024,
    )

    print("\n" + "="*80)
    print("DATASET BUILD COMPLETE")
    print("="*80)
    print(f"\nDataset saved to: {path}")
    print("\nTo inspect the dataset:")
    print("  import pandas as pd")
    print(f"  df = pd.read_parquet('{path}')")
    print("  print(df.head())")
    print("\n" + "="*80 + "\n")
