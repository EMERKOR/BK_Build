"""
Quick test script to verify data loading works with our actual files.
"""

import sys
from pathlib import Path
import importlib.util

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ball_knower.io import loaders

# Import config and team_mapping directly (avoid importing models via src/__init__.py)
_project_root = Path(__file__).parent

_config_spec = importlib.util.spec_from_file_location("config", _project_root / "src" / "config.py")
config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config)

_team_mapping_spec = importlib.util.spec_from_file_location("team_mapping", _project_root / "src" / "team_mapping.py")
team_mapping = importlib.util.module_from_spec(_team_mapping_spec)
_team_mapping_spec.loader.exec_module(team_mapping)

def test_config():
    """Test configuration."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    print(config.get_config_summary())
    assert config.PROJECT_ROOT.exists(), "Project root not found"
    assert config.DATA_DIR.exists(), "Data directory not found"
    print("✓ Config test passed\n")


def test_team_normalization():
    """Test team name normalization."""
    print("="*60)
    print("TESTING TEAM NAME NORMALIZATION")
    print("="*60 + "\n")

    # Test various formats
    test_cases = [
        ('LAR', 'LAR'),
        ('Rams', 'LAR'),
        ('Los Angeles Rams', 'LAR'),
        ('ram', 'LAR'),
        ('KC', 'KC'),
        ('Chiefs', 'KC'),
        ('kan', 'KC'),
        ('BUF', 'BUF'),
        ('buf', 'BUF'),
        ('Bills', 'BUF'),
    ]

    for input_name, expected in test_cases:
        result = team_mapping.normalize_team_name(input_name)
        assert result == expected, f"Failed: {input_name} -> {result}, expected {expected}"
        print(f"✓ '{input_name}' -> '{result}'")

    print("\n✓ Team normalization test passed\n")


def test_data_loading():
    """Test loading actual data files."""
    print("="*60)
    print("TESTING DATA LOADING (UNIFIED LOADER)")
    print("="*60 + "\n")

    # Test loading all sources via unified loader
    all_data = loaders.load_all_sources(season=2025, week=11)

    # Test nfelo power ratings
    nfelo_power = all_data['power_ratings_nfelo']
    assert len(nfelo_power) == 32, f"Expected 32 teams, got {len(nfelo_power)}"
    assert 'team' in nfelo_power.columns, "Missing 'team' column"
    print(f"✓ nfelo power ratings: {len(nfelo_power)} teams")
    print(f"  Columns: {list(nfelo_power.columns[:5])}...")

    # Test nfelo EPA tiers
    nfelo_epa = all_data['epa_tiers_nfelo']
    assert len(nfelo_epa) == 32, f"Expected 32 teams, got {len(nfelo_epa)}"
    assert 'team' in nfelo_epa.columns, "Missing 'team' column"
    print(f"✓ nfelo EPA tiers: {len(nfelo_epa)} teams")

    # Test nfelo strength of schedule
    nfelo_sos = all_data['strength_of_schedule_nfelo']
    assert len(nfelo_sos) == 32, f"Expected 32 teams, got {len(nfelo_sos)}"
    assert 'team' in nfelo_sos.columns, "Missing 'team' column"
    print(f"✓ nfelo strength of schedule: {len(nfelo_sos)} teams")

    # Test Substack power ratings
    substack_power = all_data['power_ratings_substack']
    assert len(substack_power) >= 30, f"Expected ~32 teams, got {len(substack_power)}"
    assert 'team' in substack_power.columns, "Missing 'team' column"
    print(f"✓ Substack power ratings: {len(substack_power)} teams")

    # Test Substack QB EPA
    substack_qb = all_data['qb_epa_substack']
    assert len(substack_qb) > 0, "No QBs in QB EPA data"
    assert 'team' in substack_qb.columns, "Missing 'team' column"
    print(f"✓ Substack QB EPA: {len(substack_qb)} QBs")

    # Test Substack weekly projections
    substack_weekly = all_data['weekly_projections_ppg_substack']
    assert len(substack_weekly) > 0, "No games in weekly projections"
    assert 'team' in substack_weekly.columns, "Missing 'team' column"
    print(f"✓ Substack weekly projections PPG: {len(substack_weekly)} entries")

    # Test merged ratings
    merged = all_data['merged_ratings']
    unique_teams = merged['team'].nunique()
    assert unique_teams == 32, f"Expected 32 unique teams in merged, got {unique_teams}"
    assert 'team' in merged.columns, "Missing 'team' in merged"
    print(f"✓ Merged ratings: {len(merged)} rows with {unique_teams} unique teams and {len(merged.columns)} features")
    print(f"  Sample teams: {merged['team'].unique()[:5].tolist()}")

    print("\n✓ Data loading test passed\n")


if __name__ == "__main__":
    try:
        test_config()
        test_team_normalization()
        test_data_loading()

        print("="*60)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
