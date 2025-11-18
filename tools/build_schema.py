#!/usr/bin/env python3
"""
Ball_Knower v2.0 Schema Analysis Tool

Scans all CSV files under data/ directory and automatically infers a unified dataset schema.
Generates comprehensive analysis outputs including JSON, CSV, and Markdown documentation.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
from difflib import SequenceMatcher


class SchemaAnalyzer:
    """Analyzes CSV files to infer unified schema for Ball_Knower v2.0"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.column_info = defaultdict(lambda: {
            'files': [],
            'dtypes': [],
            'missing_rates': [],
            'min_values': [],
            'max_values': [],
            'unique_counts': [],
            'sample_values': []
        })
        self.file_metadata = {}
        self.scanned_files = []

    def scan_csv_files(self) -> List[Path]:
        """Recursively scan data/ directory for all CSV files"""
        csv_pattern = str(self.data_dir / "**" / "*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
        self.scanned_files = [Path(f) for f in sorted(csv_files)]
        print(f"Found {len(self.scanned_files)} CSV files to analyze")
        return self.scanned_files

    def infer_dtype_category(self, dtype) -> str:
        """Categorize pandas dtype into simplified categories"""
        dtype_str = str(dtype)
        if 'int' in dtype_str:
            return 'integer'
        elif 'float' in dtype_str:
            return 'float'
        elif 'bool' in dtype_str:
            return 'boolean'
        elif 'datetime' in dtype_str:
            return 'datetime'
        elif 'object' in dtype_str:
            return 'string'
        else:
            return 'other'

    def analyze_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Extract comprehensive metadata for a single column"""
        metadata = {
            'name': col_name,
            'dtype': self.infer_dtype_category(series.dtype),
            'dtype_raw': str(series.dtype),
            'missing_count': int(series.isna().sum()),
            'missing_rate': float(series.isna().mean()),
            'total_count': len(series),
            'unique_count': int(series.nunique()),
        }

        # Add numeric statistics if applicable
        if metadata['dtype'] in ['integer', 'float']:
            try:
                metadata['min'] = float(series.min()) if pd.notna(series.min()) else None
                metadata['max'] = float(series.max()) if pd.notna(series.max()) else None
                metadata['mean'] = float(series.mean()) if pd.notna(series.mean()) else None
                metadata['median'] = float(series.median()) if pd.notna(series.median()) else None
            except:
                pass

        # Sample unique values (first 5)
        try:
            unique_vals = series.dropna().unique()[:5]
            metadata['sample_values'] = [str(v) for v in unique_vals]
        except:
            metadata['sample_values'] = []

        return metadata

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single CSV file and extract all column metadata"""
        rel_path = file_path.relative_to(self.data_dir)
        print(f"  Analyzing: {rel_path}")

        try:
            # Load CSV with best-effort dtype inference
            df = pd.read_csv(file_path, low_memory=False)

            # Try to infer better dtypes
            df = df.infer_objects()

            file_meta = {
                'path': str(rel_path),
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'file_size': file_path.stat().st_size,
                'column_details': {}
            }

            # Analyze each column
            for col in df.columns:
                col_meta = self.analyze_column(df[col], col)
                file_meta['column_details'][col] = col_meta

                # Aggregate column info across files
                self.column_info[col]['files'].append(str(rel_path))
                self.column_info[col]['dtypes'].append(col_meta['dtype'])
                self.column_info[col]['missing_rates'].append(col_meta['missing_rate'])
                self.column_info[col]['unique_counts'].append(col_meta['unique_count'])

                if 'min' in col_meta and col_meta['min'] is not None:
                    self.column_info[col]['min_values'].append(col_meta['min'])
                if 'max' in col_meta and col_meta['max'] is not None:
                    self.column_info[col]['max_values'].append(col_meta['max'])
                if col_meta['sample_values']:
                    self.column_info[col]['sample_values'].extend(col_meta['sample_values'])

            self.file_metadata[str(rel_path)] = file_meta
            return file_meta

        except Exception as e:
            print(f"    ERROR: Failed to analyze {rel_path}: {e}")
            return {
                'path': str(rel_path),
                'error': str(e)
            }

    def compute_canonical_schema(self) -> List[Dict[str, Any]]:
        """Compute unified schema with confidence-weighted canonical dtypes"""
        schema = []

        for col_name, info in sorted(self.column_info.items()):
            # Determine most common dtype
            dtype_counts = defaultdict(int)
            for dt in info['dtypes']:
                dtype_counts[dt] += 1

            most_common_dtype = max(dtype_counts.items(), key=lambda x: x[1])[0]
            dtype_confidence = dtype_counts[most_common_dtype] / len(info['dtypes'])

            # Check for dtype conflicts
            unique_dtypes = set(info['dtypes'])
            has_conflict = len(unique_dtypes) > 1

            # Compute stats
            avg_missing_rate = np.mean(info['missing_rates']) if info['missing_rates'] else 0

            entry = {
                'column_name': col_name,
                'detected_in_files': info['files'],
                'file_count': len(info['files']),
                'inferred_dtype': most_common_dtype,
                'dtype_confidence': round(dtype_confidence, 3),
                'dtype_conflicts': list(unique_dtypes) if has_conflict else None,
                'missing_rate_min': round(min(info['missing_rates']), 3) if info['missing_rates'] else 0,
                'missing_rate_max': round(max(info['missing_rates']), 3) if info['missing_rates'] else 0,
                'missing_rate_avg': round(avg_missing_rate, 3),
                'recommended_dtype': self.recommend_dtype(unique_dtypes, most_common_dtype),
                'notes': []
            }

            # Add numeric range if applicable
            if info['min_values'] and info['max_values']:
                entry['value_range_min'] = round(min(info['min_values']), 3)
                entry['value_range_max'] = round(max(info['max_values']), 3)

            # Add sample values (unique)
            if info['sample_values']:
                unique_samples = list(set(info['sample_values']))[:10]
                entry['sample_values'] = unique_samples

            # Add notes
            if has_conflict:
                entry['notes'].append(f"Type conflict detected: {unique_dtypes}")
            if avg_missing_rate > 0.5:
                entry['notes'].append(f"High missing rate: {avg_missing_rate:.1%}")
            if len(info['files']) == 1:
                entry['notes'].append("Only appears in one file")

            schema.append(entry)

        return schema

    def recommend_dtype(self, dtypes: set, most_common: str) -> str:
        """Recommend canonical dtype handling conflicts"""
        if len(dtypes) == 1:
            return most_common

        # If there's a mix of numeric types, recommend float
        if {'integer', 'float'}.issubset(dtypes):
            return 'float'

        # If string is in the mix, we need string to be safe
        if 'string' in dtypes:
            return 'string'

        return most_common

    def find_similar_columns(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Identify columns that might represent the same concept with different names"""
        similar_groups = []
        column_names = list(self.column_info.keys())
        processed = set()

        for i, col1 in enumerate(column_names):
            if col1 in processed:
                continue

            similar = [col1]
            for col2 in column_names[i+1:]:
                if col2 in processed:
                    continue

                # Check string similarity
                ratio = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()

                # Also check if one is substring of another
                is_substring = col1.lower() in col2.lower() or col2.lower() in col1.lower()

                if ratio >= threshold or is_substring:
                    similar.append(col2)
                    processed.add(col2)

            if len(similar) > 1:
                similar_groups.append({
                    'columns': similar,
                    'similarity_type': 'naming_variation',
                    'suggestion': f"Consider standardizing to one name"
                })
                processed.add(col1)

        return similar_groups

    def identify_deprecated_columns(self) -> List[Dict[str, Any]]:
        """Identify columns that appear unused or deprecated"""
        deprecated = []

        for col_name, info in self.column_info.items():
            avg_missing = np.mean(info['missing_rates'])
            file_count = len(info['files'])

            # Criteria for potentially deprecated
            if avg_missing > 0.9:
                deprecated.append({
                    'column_name': col_name,
                    'reason': f'Very high missing rate ({avg_missing:.1%})',
                    'files': info['files']
                })
            elif file_count == 1 and avg_missing > 0.5:
                deprecated.append({
                    'column_name': col_name,
                    'reason': f'Only in one file with {avg_missing:.1%} missing',
                    'files': info['files']
                })

        return deprecated

    def generate_markdown_report(self, schema: List[Dict], similar_cols: List[Dict],
                                deprecated: List[Dict]) -> str:
        """Generate comprehensive Markdown documentation"""
        md = []
        md.append("# Ball_Knower v2.0 Schema Proposal")
        md.append("")
        md.append("*Auto-generated by build_schema.py*")
        md.append("")
        md.append("---")
        md.append("")

        # Section 1: Summary
        md.append("## 1. Summary of Scanned Files")
        md.append("")
        md.append(f"**Total CSV files analyzed:** {len(self.scanned_files)}")
        md.append(f"**Total unique columns discovered:** {len(self.column_info)}")
        md.append(f"**Total rows across all files:** {sum(f.get('rows', 0) for f in self.file_metadata.values())}")
        md.append("")
        md.append("### Files Scanned:")
        md.append("")
        for file_path, meta in sorted(self.file_metadata.items()):
            if 'error' in meta:
                md.append(f"- `{file_path}` ⚠️ ERROR: {meta['error']}")
            else:
                md.append(f"- `{file_path}` ({meta['rows']} rows, {meta['columns']} columns)")
        md.append("")

        # Section 2: Proposed Unified Schema
        md.append("## 2. Proposed Unified Schema for v2.0")
        md.append("")
        md.append("The following schema represents all columns discovered across the dataset:")
        md.append("")
        md.append("| Column Name | Type | Files | Missing % | Notes |")
        md.append("|-------------|------|-------|-----------|-------|")

        for entry in schema:
            col_name = entry['column_name']
            dtype = entry['recommended_dtype']
            file_count = entry['file_count']
            missing = f"{entry['missing_rate_avg']:.1%}"
            notes = "; ".join(entry['notes'][:2]) if entry['notes'] else ""
            md.append(f"| `{col_name}` | {dtype} | {file_count} | {missing} | {notes} |")

        md.append("")

        # Section 3: Disambiguation Needed
        md.append("## 3. Columns Requiring Disambiguation")
        md.append("")

        conflicts = [e for e in schema if e['dtype_conflicts']]
        if conflicts:
            md.append("The following columns have type conflicts across files:")
            md.append("")
            for entry in conflicts:
                md.append(f"### `{entry['column_name']}`")
                md.append("")
                md.append(f"- **Detected types:** {', '.join(entry['dtype_conflicts'])}")
                md.append(f"- **Recommended type:** `{entry['recommended_dtype']}`")
                md.append(f"- **Appears in:** {entry['file_count']} file(s)")
                md.append("")
        else:
            md.append("✅ No type conflicts detected across files.")
            md.append("")

        if similar_cols:
            md.append("### Similar Column Names")
            md.append("")
            md.append("These columns may represent the same concept with different naming:")
            md.append("")
            for group in similar_cols:
                cols = "`, `".join(group['columns'])
                md.append(f"- `{cols}`")
                md.append(f"  - {group['suggestion']}")
            md.append("")

        # Section 4: Deprecated/Unused Columns
        md.append("## 4. Potentially Deprecated or Unused Columns")
        md.append("")

        if deprecated:
            md.append("The following columns may be deprecated or unused:")
            md.append("")
            for dep in deprecated:
                md.append(f"### `{dep['column_name']}`")
                md.append(f"- **Reason:** {dep['reason']}")
                md.append(f"- **Files:** {', '.join(dep['files'])}")
                md.append("")
        else:
            md.append("✅ No obviously deprecated columns detected.")
            md.append("")

        # Section 5: Recommendations
        md.append("## 5. Recommendations for Future Naming Conventions")
        md.append("")
        md.append("Based on the analysis, here are recommendations for Ball_Knower v2.0:")
        md.append("")
        md.append("### Naming Standards")
        md.append("")
        md.append("1. **Use snake_case consistently** - Most columns already follow this pattern")
        md.append("2. **Standardize team identifiers** - Decide on `team`, `team_code`, or `team_abbr`")
        md.append("3. **Prefix related metrics** - Group related columns (e.g., `off_*`, `def_*`)")
        md.append("4. **Avoid special characters** - Remove `X.`, `~~~`, or other non-alphanumeric prefixes")
        md.append("")
        md.append("### Type Consistency")
        md.append("")
        md.append("1. **Document expected types** - Create explicit type specifications for each column")
        md.append("2. **Handle missing data uniformly** - Decide on NaN vs empty string vs special values")
        md.append("3. **Numeric precision** - Standardize decimal places for rate/percentage columns")
        md.append("")
        md.append("### File Organization")
        md.append("")
        md.append("1. **Consistent file naming** - Use pattern: `{category}_{metric}_{season}_week_{week}.csv`")
        md.append("2. **Separate historical vs current** - Clear distinction between archives and live data")
        md.append("3. **Document file relationships** - Specify which files can be joined and on what keys")
        md.append("")
        md.append("---")
        md.append("")
        md.append(f"*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(md)

    def save_outputs(self, schema: List[Dict], similar_cols: List[Dict],
                    deprecated: List[Dict]):
        """Save all three output formats"""

        # 1. Save JSON
        json_output = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'summary': {
                'total_files': len(self.scanned_files),
                'total_columns': len(self.column_info),
                'files_scanned': [str(f.relative_to(self.data_dir)) for f in self.scanned_files]
            },
            'schema': schema,
            'similar_columns': similar_cols,
            'deprecated_columns': deprecated,
            'file_metadata': self.file_metadata
        }

        json_path = self.data_dir / "_schema_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"\n✓ JSON output saved to: {json_path}")

        # 2. Save CSV
        csv_data = []
        for entry in schema:
            csv_row = {
                'column_name': entry['column_name'],
                'detected_in_files': '; '.join(entry['detected_in_files']),
                'file_count': entry['file_count'],
                'inferred_dtype': entry['inferred_dtype'],
                'dtype_conflicts': '; '.join(entry['dtype_conflicts']) if entry['dtype_conflicts'] else '',
                'missing_rate_avg': entry['missing_rate_avg'],
                'missing_rate_min': entry['missing_rate_min'],
                'missing_rate_max': entry['missing_rate_max'],
                'recommended_dtype': entry['recommended_dtype'],
                'notes': '; '.join(entry['notes']) if entry['notes'] else ''
            }
            if 'value_range_min' in entry:
                csv_row['value_range_min'] = entry['value_range_min']
                csv_row['value_range_max'] = entry['value_range_max']
            csv_data.append(csv_row)

        df_schema = pd.DataFrame(csv_data)
        csv_path = self.data_dir / "_schema_analysis.csv"
        df_schema.to_csv(csv_path, index=False)
        print(f"✓ CSV output saved to: {csv_path}")

        # 3. Save Markdown
        md_content = self.generate_markdown_report(schema, similar_cols, deprecated)
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)
        md_path = docs_dir / "SCHEMA_PROPOSAL_v2.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"✓ Markdown report saved to: {md_path}")

        return json_path, csv_path, md_path

    def run(self):
        """Execute complete schema analysis pipeline"""
        print("=" * 60)
        print("Ball_Knower v2.0 Schema Analysis Tool")
        print("=" * 60)
        print()

        # Step 1: Scan for CSV files
        print("Step 1: Scanning for CSV files...")
        csv_files = self.scan_csv_files()
        print()

        # Step 2: Analyze each file
        print("Step 2: Analyzing CSV files...")
        for csv_file in csv_files:
            self.analyze_file(csv_file)
        print()

        # Step 3: Compute unified schema
        print("Step 3: Computing unified schema...")
        schema = self.compute_canonical_schema()
        print(f"  Discovered {len(schema)} unique columns")
        print()

        # Step 4: Find similar columns
        print("Step 4: Identifying similar columns...")
        similar_cols = self.find_similar_columns()
        print(f"  Found {len(similar_cols)} groups of similar columns")
        print()

        # Step 5: Identify deprecated columns
        print("Step 5: Identifying deprecated columns...")
        deprecated = self.identify_deprecated_columns()
        print(f"  Found {len(deprecated)} potentially deprecated columns")
        print()

        # Step 6: Save outputs
        print("Step 6: Saving outputs...")
        json_path, csv_path, md_path = self.save_outputs(schema, similar_cols, deprecated)
        print()

        # Final summary
        print("=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
        print()
        print("Summary:")
        print(f"  • Files analyzed: {len(csv_files)}")
        print(f"  • Unique columns: {len(schema)}")
        print(f"  • Type conflicts: {len([e for e in schema if e['dtype_conflicts']])}")
        print(f"  • Similar column groups: {len(similar_cols)}")
        print(f"  • Deprecated columns: {len(deprecated)}")
        print()
        print("Outputs:")
        print(f"  • {json_path}")
        print(f"  • {csv_path}")
        print(f"  • {md_path}")
        print()


def main():
    """Main entry point"""
    analyzer = SchemaAnalyzer(data_dir="data")
    analyzer.run()


if __name__ == "__main__":
    main()
