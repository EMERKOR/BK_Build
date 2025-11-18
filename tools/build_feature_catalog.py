#!/usr/bin/env python3
"""
Feature Catalog Builder for Ball_Knower v2.0

Standalone tool that scans all CSVs under data/ and produces a comprehensive
feature catalog with leakage risk tags for each column.

Usage:
    python tools/build_feature_catalog.py
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class FeatureCatalogBuilder:
    """Builds a feature catalog by analyzing all CSV files in data/"""

    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.column_info = defaultdict(lambda: {
            'source_files': set(),
            'dtypes': set(),
            'missing_rates': [],
            'example_values': set()
        })

    def scan_csv_files(self):
        """Recursively find all CSV files in data/"""
        csv_files = list(self.data_dir.rglob('*.csv'))
        print(f"Found {len(csv_files)} CSV files to analyze")
        return csv_files

    def analyze_csv(self, csv_path):
        """Analyze a single CSV file and collect column metadata"""
        try:
            # Try to read the CSV with best-effort dtype inference
            df = pd.read_csv(csv_path, low_memory=False)

            relative_path = csv_path.relative_to(self.data_dir)

            for col in df.columns:
                # Store source file
                self.column_info[col]['source_files'].add(str(relative_path))

                # Store dtype
                dtype_str = str(df[col].dtype)
                self.column_info[col]['dtypes'].add(dtype_str)

                # Calculate missing rate
                missing_rate = df[col].isna().sum() / len(df) * 100
                self.column_info[col]['missing_rates'].append(missing_rate)

                # Collect example values (up to 5 distinct)
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    samples = non_null_values.drop_duplicates().head(5).astype(str).tolist()
                    self.column_info[col]['example_values'].update(samples)

            print(f"  ✓ Analyzed {relative_path}: {len(df.columns)} columns, {len(df)} rows")
            return True

        except Exception as e:
            print(f"  ✗ Error reading {csv_path}: {e}")
            return False

    def classify_column(self, col_name):
        """
        Classify column into role and leakage_risk using heuristics.

        Returns: (role, leakage_risk, notes)
        """
        col_lower = col_name.lower()
        role = 'meta_misc'
        leakage_risk = 'unknown'
        notes = []

        # ========== ID/KEY COLUMNS ==========
        id_keywords = ['game_id', '_id', 'season', 'week', 'team', 'date', 'year']
        if any(kw in col_lower for kw in id_keywords):
            role = 'id_key'
            leakage_risk = 'low'
            notes.append('Identified as key/identifier column')

        # ========== TARGET OUTCOMES (HIGH LEAKAGE) ==========
        # These are outcomes we're trying to predict - always high leakage
        outcome_keywords = [
            'score', 'margin', 'winner', '_win', '_loss', 'result',
            'ats_result', 'total_result', 'over_under_result',
            'actual_', 'final_score', 'final_margin', 'game_result',
            '_won', '_lost', 'cover'
        ]
        if any(kw in col_lower for kw in outcome_keywords):
            # Exception: if it's just "wins" in aggregate stats, might be historical
            if 'wins' in col_lower and 'ytd' not in col_lower:
                role = 'target_outcome'
                leakage_risk = 'high'
                notes.append('Game outcome/result - prediction target')
            elif any(x in col_lower for x in ['actual', 'final', 'result', 'margin', 'cover', 'ats']):
                role = 'target_outcome'
                leakage_risk = 'high'
                notes.append('Game outcome/result - prediction target')

        # ========== PRE-GAME MARKET DATA ==========
        market_keywords = [
            'spread', 'line', 'total', 'moneyline', '_ml', 'odds',
            'implied_prob', 'open_', 'close_', 'closing_', 'opening_',
            'vegas', 'over_under', 'ou_line'
        ]
        if any(kw in col_lower for kw in market_keywords) and role == 'meta_misc':
            role = 'pre_game_market'
            leakage_risk = 'low'
            notes.append('Market line/odds - available pre-game')

        # ========== PRE-GAME TEAM STRENGTH ==========
        strength_keywords = [
            'elo', 'rating', 'power', 'qbr', 'qb_adj', 'dvoa',
            'rank', 'projection', 'proj_', 'expected', 'pythagorean',
            'strength', 'value', 'tier'
        ]
        if any(kw in col_lower for kw in strength_keywords) and role == 'meta_misc':
            # But watch out for "value" that might be post-game
            if 'actual' not in col_lower and 'final' not in col_lower:
                role = 'pre_game_team_strength'
                leakage_risk = 'low'
                notes.append('Team/player strength metric - pre-game rating')

        # ========== PRE-GAME STRUCTURAL ==========
        structure_keywords = [
            'stadium', 'surface', 'home', 'away', 'roof', 'weather',
            'playoff', 'division', 'conference', 'rest_days', 'bye',
            'dome', 'outdoor', 'temperature', 'wind', 'location'
        ]
        if any(kw in col_lower for kw in structure_keywords) and role == 'meta_misc':
            role = 'pre_game_structure'
            leakage_risk = 'low'
            notes.append('Structural/environmental factor - known pre-game')

        # ========== IN-GAME STATS (MEDIUM TO HIGH LEAKAGE) ==========
        ingame_keywords = [
            'yards', 'epa', 'success_rate', '_plays', 'pass_', 'rush_',
            'touchdown', 'turnover', 'penalty', 'first_down', 'third_down',
            'completion', 'attempt', 'sack', 'interception', 'fumble',
            'off_', 'def_', 'offense', 'defense'
        ]
        if any(kw in col_lower for kw in ingame_keywords) and role == 'meta_misc':
            # These are tricky - could be averages/projections (pre-game) or actuals (post-game)
            if any(x in col_lower for x in ['avg', 'per_game', 'projection', 'expected', 'ytd']):
                role = 'pre_game_team_strength'
                leakage_risk = 'low'
                notes.append('Historical/projected stat - safe for pre-game use')
            elif any(x in col_lower for x in ['total', '_epa_total', 'final']):
                role = 'post_game_summary'
                leakage_risk = 'high'
                notes.append('In-game stat total - only known post-game')
            else:
                role = 'in_game_stats'
                leakage_risk = 'medium'
                notes.append('In-game stat - verify timing before use')

        # ========== POST-GAME SUMMARY ==========
        postgame_keywords = [
            'final_', 'actual_', 'game_result', 'total_points',
            'combined_score', 'for_against'
        ]
        if any(kw in col_lower for kw in postgame_keywords) and role == 'meta_misc':
            role = 'post_game_summary'
            leakage_risk = 'high'
            notes.append('Post-game summary stat - high leakage risk')

        # ========== SPECIAL CASES ==========
        # "Wins" in cumulative form might be safe
        if 'wins' in col_lower and 'pythag' in col_lower:
            role = 'pre_game_team_strength'
            leakage_risk = 'low'
            notes.append('Pythagorean wins - strength metric')

        # "WoW" or "YTD" suggests historical aggregates
        if any(x in col_lower for x in ['wow', 'ytd', 'season_total', 'cumulative']):
            if role == 'in_game_stats':
                role = 'pre_game_team_strength'
                leakage_risk = 'low'
                notes.append('Historical/cumulative stat - safe for trending')

        # Default for unknown
        if role == 'meta_misc' and leakage_risk == 'unknown':
            notes.append('Could not classify - manual review needed')

        notes_str = '; '.join(notes) if notes else 'No specific pattern matched'
        return role, leakage_risk, notes_str

    def build_catalog(self):
        """Build the complete feature catalog"""
        print("\n" + "="*60)
        print("Building Feature Catalog for Ball_Knower v2.0")
        print("="*60 + "\n")

        # Scan and analyze all CSVs
        csv_files = self.scan_csv_files()
        print()

        success_count = 0
        for csv_path in sorted(csv_files):
            if self.analyze_csv(csv_path):
                success_count += 1

        print(f"\nSuccessfully analyzed {success_count}/{len(csv_files)} files")
        print(f"Found {len(self.column_info)} unique columns\n")

        # Build catalog dataframe
        catalog_rows = []

        for col_name, info in sorted(self.column_info.items()):
            role, leakage_risk, notes = self.classify_column(col_name)

            # Aggregate info
            source_files = '; '.join(sorted(info['source_files']))
            dtype_candidates = ', '.join(sorted(info['dtypes']))
            missing_rate_overall = sum(info['missing_rates']) / len(info['missing_rates'])
            example_values = ', '.join(sorted(list(info['example_values']))[:5])

            catalog_rows.append({
                'column_name': col_name,
                'source_files': source_files,
                'dtype_candidates': dtype_candidates,
                'missing_rate_overall': f"{missing_rate_overall:.2f}%",
                'role': role,
                'leakage_risk': leakage_risk,
                'example_values': example_values,
                'notes': notes
            })

        return pd.DataFrame(catalog_rows)

    def save_csv_catalog(self, df, output_path='data/_feature_catalog_raw.csv'):
        """Save the raw catalog as CSV"""
        df.to_csv(output_path, index=False)
        print(f"✓ Saved raw catalog to {output_path}")

    def generate_markdown_report(self, df, output_path='docs/FEATURE_CATALOG_v2.md'):
        """Generate comprehensive Markdown report"""

        # Ensure docs directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            # Header
            f.write("# Ball_Knower v2.0 Feature Catalog\n\n")
            f.write("**Auto-generated feature catalog and leakage risk analysis**\n\n")
            f.write("---\n\n")

            # Section 1: Summary
            f.write("## 1. Summary\n\n")
            f.write(f"- **Total unique columns analyzed:** {len(df)}\n")
            f.write(f"- **Total CSV files scanned:** {len(self.scan_csv_files())}\n")
            f.write(f"- **Data directory:** `data/`\n\n")

            # Summary by role
            f.write("### Columns by Role\n\n")
            role_counts = df['role'].value_counts()
            for role, count in role_counts.items():
                f.write(f"- **{role}:** {count}\n")
            f.write("\n")

            # Summary by leakage risk
            f.write("### Columns by Leakage Risk\n\n")
            leakage_counts = df['leakage_risk'].value_counts()
            for risk, count in leakage_counts.items():
                f.write(f"- **{risk}:** {count}\n")
            f.write("\n---\n\n")

            # Section 2: Columns by Role
            f.write("## 2. Columns by Role\n\n")

            for role in sorted(df['role'].unique()):
                role_df = df[df['role'] == role].sort_values('column_name')
                f.write(f"### {role} ({len(role_df)} columns)\n\n")
                f.write("| Column Name | Leakage Risk | Source Files | Notes |\n")
                f.write("|-------------|--------------|--------------|-------|\n")

                for _, row in role_df.iterrows():
                    sources_short = row['source_files'].split(';')[0].strip()
                    if ';' in row['source_files']:
                        sources_short += f" (+{len(row['source_files'].split(';'))-1} more)"

                    f.write(f"| `{row['column_name']}` | {row['leakage_risk']} | {sources_short} | {row['notes']} |\n")
                f.write("\n")

            f.write("---\n\n")

            # Section 3: High Leakage Risk
            f.write("## 3. High Leakage Risk Columns ⚠️\n\n")
            f.write("**These columns must be EXCLUDED from pre-game models:**\n\n")

            high_risk = df[df['leakage_risk'] == 'high'].sort_values('column_name')

            if len(high_risk) > 0:
                f.write("| Column Name | Role | Notes |\n")
                f.write("|-------------|------|-------|\n")

                for _, row in high_risk.iterrows():
                    f.write(f"| `{row['column_name']}` | {row['role']} | {row['notes']} |\n")
            else:
                f.write("*No high-risk columns identified.*\n")

            f.write("\n---\n\n")

            # Section 4: Unknown/Ambiguous
            f.write("## 4. Columns Requiring Manual Review\n\n")

            unknown = df[(df['leakage_risk'] == 'unknown') | (df['role'] == 'meta_misc')].sort_values('column_name')

            if len(unknown) > 0:
                f.write(f"**{len(unknown)} columns need manual classification:**\n\n")
                f.write("| Column Name | Current Role | Leakage Risk | Example Values | Notes |\n")
                f.write("|-------------|--------------|--------------|----------------|-------|\n")

                for _, row in unknown.iterrows():
                    examples_short = row['example_values'][:50] + '...' if len(row['example_values']) > 50 else row['example_values']
                    f.write(f"| `{row['column_name']}` | {row['role']} | {row['leakage_risk']} | {examples_short} | {row['notes']} |\n")
            else:
                f.write("*All columns successfully classified.*\n")

            f.write("\n---\n\n")

            # Section 5: Recommendations
            f.write("## 5. Recommendations for Ball_Knower v2.0\n\n")

            f.write("### ✅ SAFE for Pre-Game Models\n\n")
            f.write("Use these column types freely in v2.0:\n\n")

            safe_roles = ['id_key', 'pre_game_market', 'pre_game_team_strength', 'pre_game_structure']
            for role in safe_roles:
                count = len(df[df['role'] == role])
                f.write(f"- **{role}** ({count} columns)\n")

            f.write("\n### ⚠️ VERIFY BEFORE USE\n\n")
            medium_risk = df[df['leakage_risk'] == 'medium']
            f.write(f"These {len(medium_risk)} columns need timing verification:\n\n")
            f.write("- **in_game_stats** - Ensure these are historical aggregates, not game-specific actuals\n")

            f.write("\n### 🚫 EXCLUDE from Pre-Game Models\n\n")
            f.write(f"These {len(high_risk)} columns contain post-game information:\n\n")

            exclude_roles = df[df['leakage_risk'] == 'high']['role'].unique()
            for role in exclude_roles:
                count = len(df[(df['role'] == role) & (df['leakage_risk'] == 'high')])
                f.write(f"- **{role}** ({count} columns)\n")

            f.write("\n### 📋 Manual Review Required\n\n")
            f.write(f"Review {len(unknown)} unclassified columns (see Section 4)\n\n")

            f.write("---\n\n")
            f.write("*Generated by `tools/build_feature_catalog.py`*\n")

        print(f"✓ Saved Markdown report to {output_path}")

    def print_summary(self, df):
        """Print concise console summary"""
        print("\n" + "="*60)
        print("FEATURE CATALOG SUMMARY")
        print("="*60 + "\n")

        print("Columns by Role:")
        role_counts = df['role'].value_counts()
        for role, count in role_counts.items():
            print(f"  {role:30s} {count:4d}")

        print("\nColumns by Leakage Risk:")
        leakage_counts = df['leakage_risk'].value_counts()
        for risk, count in leakage_counts.items():
            print(f"  {risk:30s} {count:4d}")

        print("\n" + "="*60)
        print(f"Total columns analyzed: {len(df)}")
        print("="*60 + "\n")


def main():
    """Main execution"""
    builder = FeatureCatalogBuilder(data_dir='data')

    # Build catalog
    catalog_df = builder.build_catalog()

    # Save outputs
    builder.save_csv_catalog(catalog_df, output_path='data/_feature_catalog_raw.csv')
    builder.generate_markdown_report(catalog_df, output_path='docs/FEATURE_CATALOG_v2.md')

    # Print summary
    builder.print_summary(catalog_df)

    print("✓ Feature catalog generation complete!")
    print("\nOutputs:")
    print("  - data/_feature_catalog_raw.csv")
    print("  - docs/FEATURE_CATALOG_v2.md")


if __name__ == '__main__':
    main()
