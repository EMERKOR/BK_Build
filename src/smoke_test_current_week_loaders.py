"""
Simple smoke test for the unified loader (ball_knower.io.loaders).

Goal:
    - Load all current-week data using the unified loader
    - Print a concise summary for each source:
        * key name
        * shape (rows, columns)
        * sample of columns
        * head(3)

This script should NOT modify any data; it just reads and reports.
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ball_knower.io import loaders

# Import config directly without triggering src/__init__.py which imports models
import importlib.util
_config_path = _project_root / "src" / "config.py"
_spec = importlib.util.spec_from_file_location("config", _config_path)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def main() -> None:
    season = config.CURRENT_SEASON
    week = config.CURRENT_WEEK

    # Prefer the same base directory the other scripts use for current-season data
    data_dir = Path(__file__).resolve().parents[1] / "data" / "current_season"

    print(f"=== Smoke test: unified loaders for season={season}, week={week} ===")
    print(f"Data directory: {data_dir}\n")

    data = loaders.load_all_sources(
        season=season,
        week=week,
        data_dir=data_dir,
    )

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected load_all_sources to return a dict[str, DataFrame], "
            f"got {type(data)} instead."
        )

    for key, df in data.items():
        try:
            n_rows, n_cols = df.shape
        except Exception:
            print(f"--- {key}: NOT a DataFrame-like object ---")
            print(f"type: {type(df)}\n")
            continue

        print(f"--- Source: {key} ---")
        print(f"Shape: {n_rows} rows x {n_cols} columns")

        cols = list(df.columns)
        if cols:
            print("Columns (first 10):", cols[:10])
        else:
            print("Columns: (none)")

        print("Head(3):")
        print(df.head(3))
        print()

    print("=== Smoke test complete ===")


if __name__ == "__main__":
    main()
