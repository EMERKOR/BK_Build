"""
Version and Build Information Utilities

Provides consistent version banners for Ball Knower scripts.
"""

import subprocess
from pathlib import Path
from typing import Optional


def get_git_commit_hash() -> Optional[str]:
    """
    Get the current git commit hash.

    Returns:
        str: Short git commit hash (7 characters), or None if not available
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def print_version_banner(script_name: str, model_version: str = None):
    """
    Print a consistent version banner for Ball Knower scripts.

    Args:
        script_name: Name of the script (e.g., "run_weekly_predictions")
        model_version: Optional model version (e.g., "v1.2")
    """
    from ball_knower import __version__

    git_hash = get_git_commit_hash()

    print("\n" + "="*80)
    print(f"BALL KNOWER {script_name.upper().replace('_', ' ')}")
    print("="*80)

    if model_version:
        print(f"Model Version: {model_version}")

    print(f"Ball Knower Version: {__version__}")

    if git_hash:
        print(f"Git Commit: {git_hash}")

    print("="*80)
