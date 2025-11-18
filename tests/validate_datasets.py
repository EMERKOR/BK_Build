from __future__ import annotations

"""
Validation script for Ball Knower dataset pipelines.

Covers:
- ball_knower.datasets.v1_0.build_training_frame()
- ball_knower.datasets.v1_2.build_training_frame()
- ball_knower.io.loaders (nfelo, epa_team, power_team, projections)

Run from repo root with:
    python -m tests.validate_datasets
or:
    python tests/validate_datasets.py
"""

import traceback
from typing import Callable, Dict, List, Tuple

import pandas as pd

# Core imports – adjust if your package path differs
from ball_knower.datasets import v1_0, v1_2  # type: ignore

# Unified loaders are optional but strongly preferred
try:
    from ball_knower.io import loaders  # type: ignore

    HAS_UNIFIED_LOADERS = True
except ImportError:
    loaders = None  # type: ignore
    HAS_UNIFIED_LOADERS = False

# Try to pull current season/week, but treat as optional
try:
    from src import config  # type: ignore

    CURRENT_SEASON = getattr(config, "CURRENT_SEASON", None)
    CURRENT_WEEK = getattr(config, "CURRENT_WEEK", None)
except Exception:
    config = None  # type: ignore
    CURRENT_SEASON = None
    CURRENT_WEEK = None


def _header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_summary(df: pd.DataFrame, name: str, key_cols: List[str]) -> None:
    print(f"[{name}] shape: {df.shape[0]:,} rows x {df.shape[1]:,} cols")

    # Basic NA summary (top 15 columns by NA count)
    na_counts = df.isna().sum().sort_values(ascending=False)
    print(f"[{name}] top NA columns:")
    print(na_counts.head(15))

    # Check key uniqueness & missingness
    missing_keys = [c for c in key_cols if c not in df.columns]
    if missing_keys:
        print(f"[{name}] WARNING: key columns missing: {missing_keys}")
        return

    key_na_rows = df[key_cols].isna().any(axis=1).sum()
    if key_na_rows:
        print(f"[{name}] WARNING: {key_na_rows} rows have NA in key columns {key_cols}")

    dup_keys = df.duplicated(subset=key_cols).sum()
    if dup_keys:
        print(f"[{name}] WARNING: {dup_keys} duplicate key rows on {key_cols}")
    else:
        print(f"[{name}] key columns {key_cols} look unique.")


def _check_feature_stability(df: pd.DataFrame, name: str) -> None:
    """
    Confirm that build_training_frame() returns a stable feature set across seasons.
    We treat anything that is clearly an ID / target as non-feature.
    """

    if "season" not in df.columns:
        print(f"[{name}] SKIP feature stability: no 'season' column found.")
        return

    id_like = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    target_like = {
        "actual_margin",
        "target_margin",
        "home_win",
        "home_cover",
        "cover_result",
    }

    non_feature_cols = id_like | target_like
    features = [c for c in df.columns if c not in non_feature_cols]

    print(f"[{name}] total columns: {len(df.columns)}, feature columns: {len(features)}")

    # Number of columns MUST be the same per season (it always will be),
    # but we also check for features that are completely NA in some seasons.
    seasons = sorted(df["season"].unique().tolist())
    print(f"[{name}] seasons in frame: {seasons}")

    per_season_missing: Dict[int, List[str]] = {}
    for s in seasons:
        sub = df[df["season"] == s]
        all_na = sub[features].isna().all()
        missing_feats = all_na[all_na].index.tolist()
        if missing_feats:
            per_season_missing[s] = missing_feats

    if per_season_missing:
        print(f"[{name}] WARNING: some features are entirely NA in some seasons.")
        for s, feats in per_season_missing.items():
            print(f"  - Season {s}: {len(feats)} all-NA feature columns (first 10 shown): {feats[:10]}")
    else:
        print(f"[{name}] All feature columns have at least one non-NA value in every season.")


def _scan_for_leakage(df: pd.DataFrame, name: str) -> None:
    """
    Heuristic scan for columns that *might* be leaky:
    anything with obviously post-game info in the name.
    You will still need to review these manually.
    """
    leakage_keywords = [
        "final_margin",
        "actual_margin",
        "result",
        "winner",
        "loser",
        "home_score",
        "away_score",
        "total_points",
        "closing_total",
        "closing_spread",
        "postgame",
        "realized",
        "cover",
    ]

    suspicious = []
    cols_lower = {c: c.lower() for c in df.columns}
    for orig, lower in cols_lower.items():
        if any(k in lower for k in leakage_keywords):
            suspicious.append(orig)

    print(f"[{name}] potential leakage columns (needs manual review):")
    if suspicious:
        print("  " + ", ".join(sorted(set(suspicious))))
    else:
        print("  (none matched simple keyword scan)")


def validate_training_frame(builder: Callable[[], pd.DataFrame], name: str) -> pd.DataFrame:
    _header(f"Validating training frame: {name}")
    try:
        df = builder()
    except Exception as e:
        print(f"[{name}] ERROR constructing training frame:")
        print("".join(traceback.format_exception(e)))
        return pd.DataFrame()

    key_cols = ["season", "week", "home_team", "away_team"]
    if "game_id" in df.columns:
        key_cols = ["game_id"]

    _print_summary(df, name, key_cols)
    _check_feature_stability(df, name)
    _scan_for_leakage(df, name)

    return df


def validate_unified_loaders() -> None:
    _header("Validating unified loaders (ball_knower.io.loaders)")

    if not HAS_UNIFIED_LOADERS:
        print("Unified loaders module not available; skipping loader tests.")
        return

    if CURRENT_SEASON is None or CURRENT_WEEK is None:
        print("CURRENT_SEASON / CURRENT_WEEK missing; attempting calls without them.")
        season = None
        week = None
    else:
        season = CURRENT_SEASON
        week = CURRENT_WEEK
        print(f"Using CURRENT_SEASON={season}, CURRENT_WEEK={week} for loader tests.")

    def _call_loader(loader_fn: Callable, loader_name: str) -> None:
        print(f"\n--- Loader: {loader_name} ---")
        try:
            kwargs = {}
            if season is not None:
                kwargs["season"] = season
            if week is not None:
                kwargs["week"] = week

            df = loader_fn(**kwargs)  # type: ignore[arg-type]
            if not isinstance(df, pd.DataFrame):
                print(f"[{loader_name}] WARNING: returned object is not a DataFrame: {type(df)}")
                return

            key_cols = ["season", "week"]
            for col in ["home_team", "away_team", "team"]:
                if col in df.columns:
                    key_cols.append(col)

            _print_summary(df, loader_name, key_cols)
        except FileNotFoundError as e:
            print(f"[{loader_name}] FileNotFoundError: {e}")
        except Exception as e:
            print(f"[{loader_name}] ERROR:")
            print("".join(traceback.format_exception(e)))

    # These names assume loaders mirror the legacy functions without the _legacy_ prefix
    loader_specs: List[Tuple[str, Callable]] = []

    for name in [
        "load_nfelo_power_ratings",
        "load_nfelo_epa_tiers",
        "load_nfelo_qb_rankings",
        "load_nfelo_sos",
        "load_substack_power_ratings",
        "load_substack_qb_epa",
        "load_substack_weekly_projections",
    ]:
        fn = getattr(loaders, name, None)
        if fn is not None:
            loader_specs.append((name, fn))
        else:
            print(f"[loaders] NOTE: {name} not found in ball_knower.io.loaders")

    for lname, fn in loader_specs:
        _call_loader(fn, lname)


def compare_keys(df_v1_0: pd.DataFrame, df_v1_2: pd.DataFrame) -> None:
    _header("Comparing key alignment between v1_0 and v1_2 training frames")

    if df_v1_0.empty or df_v1_2.empty:
        print("One or both frames are empty; skipping key comparison.")
        return

    key_cols = []
    for col in ["game_id", "season", "week", "home_team", "away_team"]:
        if col in df_v1_0.columns and col in df_v1_2.columns:
            key_cols.append(col)

    if not key_cols:
        print("No common key columns between v1_0 and v1_2; cannot compare keys.")
        return

    print(f"Using key columns for comparison: {key_cols}")

    k0 = df_v1_0[key_cols].drop_duplicates()
    k2 = df_v1_2[key_cols].drop_duplicates()

    merged = k0.merge(k2, on=key_cols, how="outer", indicator=True)

    only_v1_0 = merged[merged["_merge"] == "left_only"]
    only_v1_2 = merged[merged["_merge"] == "right_only"]

    print(f"Total unique keys in v1_0: {len(k0)}")
    print(f"Total unique keys in v1_2: {len(k2)}")
    print(f"Keys only in v1_0: {len(only_v1_0)}")
    print(f"Keys only in v1_2: {len(only_v1_2)}")

    if len(only_v1_0) > 0:
        print("Example keys only in v1_0 (first 10):")
        print(only_v1_0.head(10))

    if len(only_v1_2) > 0:
        print("Example keys only in v1_2 (first 10):")
        print(only_v1_2.head(10))


def main() -> None:
    # 1) Unified loaders for nfelo / epa_team / power_team / projections
    validate_unified_loaders()

    # 2) v1_0 training frame
    df_v1_0 = validate_training_frame(v1_0.build_training_frame, "v1_0")

    # 3) v1_2 training frame
    df_v1_2 = validate_training_frame(v1_2.build_training_frame, "v1_2")

    # 4) Cross-compare keys between v1_0 and v1_2
    compare_keys(df_v1_0, df_v1_2)


if __name__ == "__main__":
    main()
