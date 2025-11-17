"""
Quick test script to verify data loading works with our actual files.
Updated for Phase 3: Uses unified loaders from ball_knower.io
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config, team_mapping
from ball_knower.io import loaders

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
    """Test loading actual data files using unified loaders."""
    print("="*60)
    print("TESTING DATA LOADING (Unified Loaders)")
    print("="*60 + "\n")

    # Test loading nfelo power ratings
    nfelo_power = loaders.load_power_ratings('nfelo', config.CURRENT_SEASON, config.CURRENT_WEEK, './data')
    assert nfelo_power is not None, "Failed to load nfelo power ratings"
    assert len(nfelo_power) == 32, f"Expected 32 teams, got {len(nfelo_power)}"
    assert 'team' in nfelo_power.columns, "Missing 'team' column"
    print(f"✓ nfelo power ratings: {len(nfelo_power)} teams")
    print(f"  Columns: {list(nfelo_power.columns[:5])}...")

    # Test loading nfelo EPA tiers
    nfelo_epa = loaders.load_epa_tiers('nfelo', config.CURRENT_SEASON, config.CURRENT_WEEK, './data')
    assert nfelo_epa is not None, "Failed to load nfelo EPA tiers"
    assert len(nfelo_epa) == 32, f"Expected 32 teams, got {len(nfelo_epa)}"
    assert 'epa_off' in nfelo_epa.columns, "Missing 'epa_off' column"
    assert 'epa_def' in nfelo_epa.columns, "Missing 'epa_def' column"
    print(f"✓ nfelo EPA tiers: {len(nfelo_epa)} teams")

    # Test loading Substack power ratings
    substack_power = loaders.load_power_ratings('substack', config.CURRENT_SEASON, config.CURRENT_WEEK, './data')
    assert substack_power is not None, "Failed to load substack power ratings"
    assert len(substack_power) >= 30, f"Expected ~32 teams, got {len(substack_power)}"
    assert 'team' in substack_power.columns, "Missing 'team' column"
    print(f"✓ Substack power ratings: {len(substack_power)} teams")

    # Test loading Substack weekly projections
    substack_weekly = loaders.load_weekly_projections_ppg('substack', config.CURRENT_SEASON, config.CURRENT_WEEK, './data')
    assert substack_weekly is not None, "Failed to load substack weekly projections"
    assert len(substack_weekly) > 0, "No games in weekly projections"
    print(f"✓ Substack weekly projections: {len(substack_weekly)} games")

    # Test load_all_sources and merged ratings
    data = loaders.load_all_sources(week=config.CURRENT_WEEK, season=config.CURRENT_SEASON, data_dir='./data')
    assert 'merged_ratings' in data, "Missing merged_ratings in data"

    merged = data['merged_ratings']
    assert len(merged) == 32, f"Expected 32 teams in merged, got {len(merged)}"
    assert 'nfelo' in merged.columns, "Missing nfelo in merged"
    assert 'epa_off' in merged.columns, "Missing epa_off in merged"
    print(f"✓ Merged ratings: {len(merged)} teams with {len(merged.columns)} features")
    print(f"  Sample teams: {merged['team'].head(5).tolist()}")

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
