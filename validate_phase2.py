"""
Validation script for Phase 2 compatibility layer.
This script checks the structure without requiring pandas or ball_knower.io.loaders.
"""

import ast
import inspect
from pathlib import Path

def analyze_data_loader():
    """Analyze the refactored data_loader.py structure."""

    print("="*60)
    print("PHASE 2 VALIDATION - Compatibility Layer Structure")
    print("="*60 + "\n")

    # Read the refactored file
    data_loader_path = Path("/home/user/BK_Build/src/data_loader.py")
    content = data_loader_path.read_text()

    # Parse the AST
    tree = ast.parse(content)

    # Extract function names
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    # Check for required patterns
    checks = {
        "NEW_LOADERS_AVAILABLE flag": "NEW_LOADERS_AVAILABLE" in content,
        "Import try/except block": "from ball_knower.io import loaders as new_loaders" in content,
        "Legacy power ratings": "_legacy_load_nfelo_power_ratings" in functions,
        "Legacy EPA tiers": "_legacy_load_nfelo_epa_tiers" in functions,
        "Legacy QB rankings": "_legacy_load_nfelo_qb_rankings" in functions,
        "Legacy SOS": "_legacy_load_nfelo_sos" in functions,
        "Legacy Substack power": "_legacy_load_substack_power_ratings" in functions,
        "Legacy Substack QB EPA": "_legacy_load_substack_qb_epa" in functions,
        "Legacy weekly projections": "_legacy_load_substack_weekly_projections" in functions,
        "Legacy load_all": "_legacy_load_all_current_week_data" in functions,
        "Legacy merge": "_legacy_merge_current_week_ratings" in functions,
        "Public load_nfelo_power_ratings": "load_nfelo_power_ratings" in functions,
        "Public load_nfelo_epa_tiers": "load_nfelo_epa_tiers" in functions,
        "Public load_nfelo_qb_rankings": "load_nfelo_qb_rankings" in functions,
        "Public load_nfelo_sos": "load_nfelo_sos" in functions,
        "Public load_substack_power_ratings": "load_substack_power_ratings" in functions,
        "Public load_substack_qb_epa": "load_substack_qb_epa" in functions,
        "Public load_substack_weekly_projections": "load_substack_weekly_projections" in functions,
        "Public load_all_current_week_data": "load_all_current_week_data" in functions,
        "Public merge_current_week_ratings": "merge_current_week_ratings" in functions,
        "Deprecation warnings": content.count("DeprecationWarning") >= 9,
        "Fallback to legacy": content.count("return _legacy_") >= 9,
        "NEW_LOADERS_AVAILABLE checks": content.count("if NEW_LOADERS_AVAILABLE:") >= 9,
    }

    # Print results
    all_passed = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("✓✓✓ PHASE 2 STRUCTURE VALIDATION PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
    print(f"{'='*60}\n")

    # Print function summary
    print("FUNCTION SUMMARY:")
    print(f"  Total functions: {len(functions)}")
    print(f"  Legacy functions (_legacy_*): {len([f for f in functions if f.startswith('_legacy_')])}")
    print(f"  Public loader functions: {len([f for f in functions if f.startswith('load_') and not f.startswith('_legacy_')])}")

    # Show the compatibility pattern for one function
    print(f"\n{'='*60}")
    print("EXAMPLE COMPATIBILITY PATTERN (load_nfelo_power_ratings):")
    print(f"{'='*60}")

    # Extract the function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_nfelo_power_ratings":
            # Get the source lines
            start_line = node.lineno
            end_line = node.end_lineno
            lines = content.split('\n')[start_line-1:end_line]
            print('\n'.join(lines[:25]))  # Show first 25 lines
            if len(lines) > 25:
                print("...")
            break

    return all_passed

if __name__ == "__main__":
    success = analyze_data_loader()
    exit(0 if success else 1)
