#!/usr/bin/env python3
"""
Ball Knower v2.0 - Unified Data Dictionary Generator

Merges outputs from three analysis tools:
- data/_schema_analysis.json (schema metadata per column)
- data/_feature_catalog_raw.csv (role + leakage tags per column)
- data/_file_mapping_preview.csv (file-level mapping and context)

Produces:
- data/_data_dictionary_v2.csv (machine-readable)
- docs/DATA_DICTIONARY_v2.md (human-readable)
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


# File paths
SCHEMA_ANALYSIS_PATH = Path("data/_schema_analysis.json")
FEATURE_CATALOG_PATH = Path("data/_feature_catalog_raw.csv")
FILE_MAPPING_PATH = Path("data/_file_mapping_preview.csv")
OUTPUT_CSV_PATH = Path("data/_data_dictionary_v2.csv")
OUTPUT_MD_PATH = Path("docs/DATA_DICTIONARY_v2.md")


def check_input_files() -> bool:
    """Verify all required input files exist."""
    missing = []
    for path in [SCHEMA_ANALYSIS_PATH, FEATURE_CATALOG_PATH, FILE_MAPPING_PATH]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        print("ERROR: Missing required input files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease run the analysis tools first:")
        print("  1. Schema analysis tool -> data/_schema_analysis.json")
        print("  2. Feature catalog tool -> data/_feature_catalog_raw.csv")
        print("  3. File mapping tool -> data/_file_mapping_preview.csv")
        return False
    return True


def load_schema_analysis() -> Dict[str, Any]:
    """Load schema analysis JSON."""
    with open(SCHEMA_ANALYSIS_PATH, 'r') as f:
        return json.load(f)


def load_feature_catalog() -> List[Dict[str, str]]:
    """Load feature catalog CSV."""
    with open(FEATURE_CATALOG_PATH, 'r') as f:
        return list(csv.DictReader(f))


def load_file_mapping() -> List[Dict[str, str]]:
    """Load file mapping CSV."""
    with open(FILE_MAPPING_PATH, 'r') as f:
        return list(csv.DictReader(f))


def merge_data(schema_data: Dict, feature_catalog: List[Dict], file_mapping: List[Dict]) -> List[Dict]:
    """
    Merge all three data sources at the column level.

    Returns a list of dictionaries, one per column, containing:
    - column_name
    - source_files
    - logical_datasets
    - recommended_dtype
    - missing_rate_overall
    - role
    - leakage_risk
    - example_values
    - conflicts
    - notes
    """

    # Build file mapping lookup: filename -> logical_dataset, provider, category
    file_to_logical = {}
    for row in file_mapping:
        filename = row.get('filename', row.get('file', ''))
        file_to_logical[filename] = {
            'logical_dataset': row.get('logical_dataset', row.get('dataset', '')),
            'provider': row.get('provider', ''),
            'category': row.get('category', '')
        }

    # Build feature catalog lookup: column_name -> data
    feature_lookup = {}
    for row in feature_catalog:
        col_name = row.get('column_name', '')
        if col_name:
            feature_lookup[col_name] = row

    # Process schema analysis (main source of columns)
    merged = []

    # Handle different possible schema structures
    columns_data = schema_data.get('columns', {})
    if isinstance(columns_data, dict):
        # Dictionary format: column_name -> metadata
        for col_name, col_info in columns_data.items():
            merged.append(merge_column(col_name, col_info, feature_lookup, file_to_logical))
    elif isinstance(columns_data, list):
        # List format
        for col_info in columns_data:
            col_name = col_info.get('column_name', col_info.get('name', ''))
            if col_name:
                merged.append(merge_column(col_name, col_info, feature_lookup, file_to_logical))

    # Also check for any columns in feature catalog not in schema
    schema_cols = {item['column_name'] for item in merged}
    for col_name, feat_data in feature_lookup.items():
        if col_name not in schema_cols:
            # Column in feature catalog but not in schema
            merged.append(merge_column(col_name, {}, feature_lookup, file_to_logical))

    return merged


def merge_column(col_name: str, schema_info: Dict, feature_lookup: Dict,
                 file_to_logical: Dict) -> Dict:
    """Merge information for a single column from all sources."""

    # Get feature catalog data
    feat_data = feature_lookup.get(col_name, {})

    # Extract source files
    source_files = feat_data.get('source_files', schema_info.get('source_files', ''))

    # Determine logical datasets from source files
    logical_datasets = []
    if source_files:
        for file in source_files.split(';'):
            file = file.strip()
            if file in file_to_logical:
                logical_dataset = file_to_logical[file]['logical_dataset']
                if logical_dataset and logical_dataset not in logical_datasets:
                    logical_datasets.append(logical_dataset)

    # Extract data types
    inferred_dtype = schema_info.get('inferred_dtype', schema_info.get('dtype', ''))
    recommended_dtype = schema_info.get('recommended_dtype', inferred_dtype)

    # Extract missing rate
    missing_rate = schema_info.get('missing_rate_overall',
                                   schema_info.get('missing_rate', ''))

    # Extract role and leakage
    role = feat_data.get('role', schema_info.get('role', 'unknown'))
    leakage_risk = feat_data.get('leakage_risk', schema_info.get('leakage_risk', 'unknown'))

    # Extract example values
    example_values = schema_info.get('example_values', '')
    if isinstance(example_values, list):
        example_values = ', '.join(str(v) for v in example_values[:5])

    # Extract conflicts
    conflicts = schema_info.get('conflicts', schema_info.get('conflict', ''))
    if isinstance(conflicts, list):
        conflicts = '; '.join(conflicts)

    # Extract notes
    notes = feat_data.get('notes', schema_info.get('notes', ''))

    return {
        'column_name': col_name,
        'source_files': source_files,
        'logical_datasets': '; '.join(logical_datasets),
        'recommended_dtype': recommended_dtype,
        'missing_rate_overall': missing_rate,
        'role': role,
        'leakage_risk': leakage_risk,
        'example_values': example_values,
        'conflicts': conflicts,
        'notes': notes
    }


def write_csv(data: List[Dict]) -> None:
    """Write machine-readable CSV output."""
    if not data:
        print("WARNING: No data to write to CSV")
        return

    fieldnames = [
        'column_name', 'source_files', 'logical_datasets', 'recommended_dtype',
        'missing_rate_overall', 'role', 'leakage_risk', 'example_values',
        'conflicts', 'notes'
    ]

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"✓ Generated: {OUTPUT_CSV_PATH}")


def write_markdown(data: List[Dict]) -> None:
    """Write human-readable Markdown output with structured sections."""

    if not data:
        print("WARNING: No data to write to Markdown")
        return

    # Group data by role and leakage
    by_role = defaultdict(list)
    by_leakage = defaultdict(list)
    for item in data:
        by_role[item['role']].append(item)
        by_leakage[item['leakage_risk']].append(item)

    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD_PATH, 'w') as f:
        f.write("# Ball Knower v2.0 - Data Dictionary\n\n")
        f.write("*Auto-generated from schema analysis, feature catalog, and file mapping tools*\n\n")

        # Section 1: Overview
        f.write("## 1. Overview\n\n")
        unique_files = set()
        for item in data:
            if item['source_files']:
                for file in item['source_files'].split(';'):
                    unique_files.add(file.strip())

        f.write(f"- **Total Columns**: {len(data)}\n")
        f.write(f"- **Source Files**: {len(unique_files)}\n")
        f.write(f"- **Roles Distribution**:\n")
        for role in sorted(by_role.keys()):
            f.write(f"  - `{role}`: {len(by_role[role])} columns\n")
        f.write(f"- **Leakage Risk Distribution**:\n")
        for leakage in ['low', 'medium', 'high', 'unknown']:
            count = len(by_leakage.get(leakage, []))
            if count > 0:
                f.write(f"  - `{leakage}`: {count} columns\n")
        f.write("\n")

        # Section 2: Core ID/Key Columns
        f.write("## 2. Core ID/Key Columns\n\n")
        id_cols = by_role.get('id_key', [])
        if id_cols:
            f.write("| Column Name | Source Files | Dtype | Notes |\n")
            f.write("|-------------|--------------|-------|-------|\n")
            for item in sorted(id_cols, key=lambda x: x['column_name']):
                f.write(f"| `{item['column_name']}` | {item['source_files'][:50]}... | "
                       f"{item['recommended_dtype']} | {item['notes'][:50]} |\n")
        else:
            f.write("*No ID/Key columns identified*\n")
        f.write("\n")

        # Section 3: Safe Pre-Game Features
        f.write("## 3. Safe Pre-Game Features\n\n")
        f.write("*Columns with `leakage_risk = low`, safe for pre-game modeling*\n\n")

        safe_cols = by_leakage.get('low', [])
        if safe_cols:
            # Group by role
            safe_by_role = defaultdict(list)
            for item in safe_cols:
                safe_by_role[item['role']].append(item)

            for role in sorted(safe_by_role.keys()):
                f.write(f"### {role.replace('_', ' ').title()}\n\n")
                f.write("| Column Name | Dtype | Missing % | Notes |\n")
                f.write("|-------------|-------|-----------|-------|\n")
                for item in sorted(safe_by_role[role], key=lambda x: x['column_name']):
                    missing = item['missing_rate_overall']
                    if missing and isinstance(missing, (int, float)):
                        missing = f"{float(missing):.1%}"
                    f.write(f"| `{item['column_name']}` | {item['recommended_dtype']} | "
                           f"{missing} | {item['notes'][:60]} |\n")
                f.write("\n")
        else:
            f.write("*No safe pre-game features identified*\n\n")

        # Section 4: High / Medium Leakage Features
        f.write("## 4. High / Medium Leakage Features\n\n")
        f.write("*Columns that may contain in-game or post-game information*\n\n")

        risky_cols = by_leakage.get('high', []) + by_leakage.get('medium', [])
        if risky_cols:
            f.write("| Column Name | Leakage Risk | Role | Reason |\n")
            f.write("|-------------|--------------|------|--------|\n")
            for item in sorted(risky_cols, key=lambda x: (x['leakage_risk'], x['column_name'])):
                f.write(f"| `{item['column_name']}` | **{item['leakage_risk']}** | "
                       f"{item['role']} | {item['notes'][:80]} |\n")
        else:
            f.write("*No high/medium leakage features identified*\n")
        f.write("\n")

        # Section 5: Ambiguous / Unknown Columns
        f.write("## 5. Ambiguous / Unknown Columns\n\n")
        f.write("*Columns requiring manual review*\n\n")

        unknown_cols = [item for item in data
                       if item['role'] in ['meta_misc', 'unknown']
                       or item['leakage_risk'] == 'unknown']
        if unknown_cols:
            f.write("| Column Name | Role | Leakage Risk | Source Files | Notes |\n")
            f.write("|-------------|------|--------------|--------------|-------|\n")
            for item in sorted(unknown_cols, key=lambda x: x['column_name']):
                f.write(f"| `{item['column_name']}` | {item['role']} | "
                       f"{item['leakage_risk']} | {item['source_files'][:30]}... | "
                       f"{item['notes'][:50]} |\n")
        else:
            f.write("*No ambiguous columns - all columns have been classified*\n")
        f.write("\n")

        # Section 6: Recommendations
        f.write("## 6. Recommendations\n\n")

        f.write("### Safe for v2.0 Pre-Game Models\n\n")
        safe_count = len(by_leakage.get('low', []))
        f.write(f"- **{safe_count} columns** identified as safe (low leakage risk)\n")
        f.write("- Focus on columns with roles: `structure`, `pre_game_market`, `pre_game_team_strength`\n")
        if safe_cols:
            f.write("- Key safe features:\n")
            for item in sorted(safe_cols, key=lambda x: x['column_name'])[:10]:
                f.write(f"  - `{item['column_name']}`\n")
        f.write("\n")

        f.write("### Must Exclude from Pre-Game Models\n\n")
        high_risk = by_leakage.get('high', [])
        f.write(f"- **{len(high_risk)} columns** with high leakage risk must be excluded\n")
        if high_risk:
            f.write("- High-risk columns to avoid:\n")
            for item in sorted(high_risk, key=lambda x: x['column_name'])[:10]:
                f.write(f"  - `{item['column_name']}` - {item['notes'][:60]}\n")
        f.write("\n")

        f.write("### Require Manual Review\n\n")
        medium_risk = by_leakage.get('medium', [])
        unknown_risk = by_leakage.get('unknown', [])
        f.write(f"- **{len(medium_risk)} columns** with medium leakage risk need timing verification\n")
        f.write(f"- **{len(unknown_risk)} columns** with unknown risk need classification\n")
        if medium_risk:
            f.write("- Medium-risk columns to verify:\n")
            for item in sorted(medium_risk, key=lambda x: x['column_name'])[:10]:
                f.write(f"  - `{item['column_name']}` - verify timing: {item['notes'][:50]}\n")
        f.write("\n")

    print(f"✓ Generated: {OUTPUT_MD_PATH}")


def print_summary(data: List[Dict]) -> None:
    """Print concise summary to console."""
    print("\n" + "="*60)
    print("DATA DICTIONARY GENERATION SUMMARY")
    print("="*60)

    # Count by role
    by_role = defaultdict(int)
    by_leakage = defaultdict(int)
    for item in data:
        by_role[item['role']] += 1
        by_leakage[item['leakage_risk']] += 1

    print(f"\nTotal Columns: {len(data)}")

    print("\nRole Distribution:")
    for role in sorted(by_role.keys()):
        print(f"  {role:30s}: {by_role[role]:4d}")

    print("\nLeakage Risk Distribution:")
    for leakage in ['low', 'medium', 'high', 'unknown']:
        count = by_leakage.get(leakage, 0)
        print(f"  {leakage:30s}: {count:4d}")

    print("\n" + "="*60)
    print(f"Outputs written to:")
    print(f"  - {OUTPUT_CSV_PATH}")
    print(f"  - {OUTPUT_MD_PATH}")
    print("="*60 + "\n")


def main():
    """Main execution function."""
    print("Ball Knower v2.0 - Data Dictionary Generator")
    print("=" * 60)

    # Check input files
    if not check_input_files():
        sys.exit(1)

    print("\n✓ All input files found")

    # Load data
    print("\nLoading data sources...")
    schema_data = load_schema_analysis()
    feature_catalog = load_feature_catalog()
    file_mapping = load_file_mapping()
    print(f"  - Schema analysis: {len(schema_data.get('columns', {}))} items")
    print(f"  - Feature catalog: {len(feature_catalog)} rows")
    print(f"  - File mapping: {len(file_mapping)} rows")

    # Merge data
    print("\nMerging data at column level...")
    merged_data = merge_data(schema_data, feature_catalog, file_mapping)
    print(f"  - Merged: {len(merged_data)} columns")

    # Write outputs
    print("\nGenerating outputs...")
    write_csv(merged_data)
    write_markdown(merged_data)

    # Print summary
    print_summary(merged_data)

    print("✓ Data dictionary generation complete!")


if __name__ == "__main__":
    main()
