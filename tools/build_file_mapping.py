#!/usr/bin/env python3
"""
File Naming Analysis and Mapping Tool

This script analyzes all CSV files under data/ directory and generates a structured
mapping from current filenames to the new category-first naming convention.

Output Format: {category}_{stat_type}_{season}_week_{week}.csv
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json


class FileNameAnalyzer:
    """Analyzes football data filenames and proposes standardized names."""

    # Known categories and their variations
    CATEGORIES = {
        'offense': ['offense', 'offensive', 'off'],
        'defense': ['defense', 'defensive', 'def'],
        'qb': ['qb', 'quarterback', 'qbs'],
        'team_ratings': ['team_ratings', 'ratings', 'power_ratings', 'power'],
        'projections': ['projections', 'proj', 'projected'],
        'epa': ['epa', 'expected_points'],
        'elo': ['elo', 'nfelo'],
        'schedule': ['schedule', 'schedules'],
        'injuries': ['injuries', 'injury'],
        'receiving': ['receiving', 'receivers'],
        'rushing': ['rushing', 'rushers'],
        'passing': ['passing', 'passers'],
        'win_totals': ['win_totals', 'wins'],
        'sos': ['sos', 'strength_of_schedule'],
        'tiers': ['tiers', 'tier'],
        'coaches': ['coaches', 'coach', 'head_coaches'],
    }

    # Known providers
    PROVIDERS = ['nfelo', '538', 'fivethirtyeight', 'substack', 'pfr', 'espn', 'nfl']

    # Known stat types
    STAT_TYPES = [
        'epa', 'power', 'elo', 'ratings', 'projections', 'rankings',
        'leaders', 'totals', 'tiers', 'ppg', 'weekly', 'season'
    ]

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.results: List[Dict] = []

    def scan_directory(self) -> List[Dict]:
        """Scan all CSV files in the data directory."""
        csv_files = list(self.data_root.rglob("*.csv"))

        for file_path in sorted(csv_files):
            analysis = self.analyze_file(file_path)
            self.results.append(analysis)

        return self.results

    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single file and extract structured information."""
        relative_path = file_path.relative_to(self.data_root)
        filename = file_path.name
        stem = file_path.stem  # filename without extension

        # Extract components
        category = self._extract_category(stem)
        stat_type = self._extract_stat_type(stem)
        provider = self._extract_provider(stem)
        season = self._extract_season(stem)
        week = self._extract_week(stem)

        # Generate proposed new filename
        proposed_name, confidence = self._generate_proposed_name(
            category, stat_type, provider, season, week, filename
        )

        # Generate notes
        notes = self._generate_notes(stem, category, stat_type, provider, season, week)

        return {
            'old_filename': filename,
            'old_path': str(relative_path),
            'inferred_category': category,
            'inferred_stat_type': stat_type,
            'inferred_provider': provider,
            'inferred_season': season,
            'inferred_week': week,
            'proposed_new_filename': proposed_name,
            'confidence_score': confidence,
            'notes': notes
        }

    def _extract_category(self, stem: str) -> str:
        """Extract category from filename."""
        stem_lower = stem.lower()

        # Special case patterns
        if 'qb' in stem_lower:
            if 'ranking' in stem_lower:
                return 'qb_rankings'
            return 'qb'

        if 'receiving' in stem_lower:
            return 'receiving'

        if 'power_rating' in stem_lower or 'power' in stem_lower and 'rating' in stem_lower:
            return 'team_ratings'

        if 'strength_of_schedule' in stem_lower or 'sos' in stem_lower:
            return 'sos'

        if 'epa_tier' in stem_lower:
            return 'epa_tiers'

        if 'win_total' in stem_lower:
            return 'win_totals'

        if 'projection' in stem_lower or 'proj' in stem_lower:
            if 'elo' in stem_lower:
                return 'elo_projections'
            if 'ppg' in stem_lower:
                return 'ppg_projections'
            return 'projections'

        if 'schedule' in stem_lower:
            return 'schedule'

        if 'coach' in stem_lower:
            return 'reference_coaches'

        if 'team' in stem_lower and 'week' in stem_lower and 'epa' in stem_lower:
            return 'team_epa'

        # Check against known categories
        for category, variations in self.CATEGORIES.items():
            for variant in variations:
                if variant in stem_lower:
                    return category

        return 'unknown'

    def _extract_stat_type(self, stem: str) -> str:
        """Extract stat type from filename."""
        stem_lower = stem.lower()

        if 'epa' in stem_lower:
            return 'epa'
        if 'elo' in stem_lower:
            return 'elo'
        if 'power' in stem_lower or 'rating' in stem_lower:
            return 'power'
        if 'ranking' in stem_lower:
            return 'rankings'
        if 'projection' in stem_lower or 'proj' in stem_lower:
            return 'projections'
        if 'leader' in stem_lower:
            return 'leaders'
        if 'tier' in stem_lower:
            return 'tiers'
        if 'schedule' in stem_lower:
            return 'schedule'
        if 'total' in stem_lower:
            return 'totals'
        if 'ppg' in stem_lower:
            return 'ppg'

        return 'general'

    def _extract_provider(self, stem: str) -> Optional[str]:
        """Extract provider from filename."""
        stem_lower = stem.lower()

        for provider in self.PROVIDERS:
            if provider in stem_lower:
                return provider

        return None

    def _extract_season(self, stem: str) -> Optional[str]:
        """Extract season/year from filename."""
        # Look for 4-digit years (2000-2099)
        # Use lookahead/lookbehind to ensure we match years but not other numbers
        year_pattern = r'(?:^|_)(20\d{2})(?:_|$|-)'
        matches = re.findall(year_pattern, stem)

        if matches:
            if len(matches) == 1:
                return matches[0]
            elif len(matches) == 2:
                # Might be a range like 2013_2024
                return f"{matches[0]}_{matches[1]}"
            else:
                return matches[0]  # Take the first one

        return None

    def _extract_week(self, stem: str) -> Optional[str]:
        """Extract week number from filename."""
        # Look for "week_XX" or "week XX" patterns
        week_pattern = r'week[_\s]+(\d{1,2})'
        match = re.search(week_pattern, stem.lower())

        if match:
            return match.group(1)

        return None

    def _generate_proposed_name(
        self, category: str, stat_type: str, provider: Optional[str],
        season: Optional[str], week: Optional[str], original: str
    ) -> Tuple[str, float]:
        """Generate proposed new filename following the standard convention."""

        confidence = 1.0
        parts = []

        # Start with category
        if category and category != 'unknown':
            parts.append(category)
        else:
            parts.append('unknown')
            confidence -= 0.3

        # Add stat type if different from category
        if stat_type and stat_type != 'general' and stat_type not in category:
            parts.append(stat_type)

        # Add provider if available
        if provider:
            parts.append(provider)

        # Add season
        if season:
            parts.append(season)
        else:
            parts.append('SEASON')
            confidence -= 0.2

        # Add week if available
        if week:
            parts.append(f'week_{week}')
        else:
            # Check if this is a reference/static file
            if 'reference' in category or 'coach' in original.lower() or 'schedule' in category:
                # Reference files don't need week
                confidence -= 0.1
            else:
                parts.append('week_XX')
                confidence -= 0.2

        proposed = '_'.join(parts) + '.csv'

        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))

        return proposed, confidence

    def _generate_notes(
        self, stem: str, category: str, stat_type: str,
        provider: Optional[str], season: Optional[str], week: Optional[str]
    ) -> str:
        """Generate notes about the analysis."""
        notes = []

        if category == 'unknown':
            notes.append("Could not confidently determine category")

        if not provider:
            notes.append("No provider identified in filename")

        if not season:
            notes.append("No season/year found")

        if not week:
            if 'reference' not in category and 'schedule' not in category:
                notes.append("No week number found - may be season-level data")

        if '(1)' in stem or '(2)' in stem:
            notes.append("Filename contains copy marker - possible duplicate")

        if '_' in stem and '-' in stem:
            notes.append("Mixed delimiter usage in original filename")

        if not notes:
            notes.append("Clear naming structure detected")

        return '; '.join(notes)

    def generate_report(self) -> pd.DataFrame:
        """Generate a pandas DataFrame report."""
        if not self.results:
            self.scan_directory()

        df = pd.DataFrame(self.results)
        return df

    def save_report(self, output_path: str = "data/_file_mapping_preview.csv"):
        """Save the report to CSV."""
        df = self.generate_report()
        df.to_csv(output_path, index=False)
        print(f"\n✓ Mapping saved to: {output_path}")
        return df


def main():
    """Main execution function."""
    print("=" * 80)
    print("FILE NAMING ANALYSIS AND MAPPING TOOL")
    print("=" * 80)
    print("\nScanning data/ directory for CSV files...\n")

    # Initialize analyzer
    analyzer = FileNameAnalyzer(data_root="data")

    # Scan and analyze
    results = analyzer.scan_directory()

    print(f"Found {len(results)} CSV files\n")
    print("-" * 80)

    # Generate and display report
    df = analyzer.generate_report()

    # Display summary statistics
    print("\nSUMMARY STATISTICS")
    print("-" * 80)
    print(f"Total files analyzed: {len(df)}")
    print(f"Average confidence score: {df['confidence_score'].mean():.2f}")
    print(f"Files with high confidence (>0.8): {(df['confidence_score'] > 0.8).sum()}")
    print(f"Files with low confidence (<0.6): {(df['confidence_score'] < 0.6).sum()}")
    print(f"\nUnique categories identified: {df['inferred_category'].nunique()}")
    print(f"Files with provider info: {df['inferred_provider'].notna().sum()}")
    print(f"Files with season info: {df['inferred_season'].notna().sum()}")
    print(f"Files with week info: {df['inferred_week'].notna().sum()}")

    # Display category breakdown
    print("\n\nCATEGORY BREAKDOWN")
    print("-" * 80)
    category_counts = df['inferred_category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

    # Display full mapping table
    print("\n\nDETAILED MAPPING TABLE")
    print("=" * 80)

    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    print(df.to_string(index=False))

    # Save to CSV
    print("\n" + "=" * 80)
    output_path = "data/_file_mapping_preview.csv"
    analyzer.save_report(output_path)

    # Also save as JSON for programmatic use
    json_path = "data/_file_mapping_preview.json"
    df.to_json(json_path, orient='records', indent=2)
    print(f"✓ Mapping also saved as JSON to: {json_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the mapping in data/_file_mapping_preview.csv")
    print("  2. Adjust confidence scores or proposed names as needed")
    print("  3. Use this mapping to rename files systematically")
    print("\nNote: No files have been renamed. This is analysis only.")
    print("=" * 80)


if __name__ == "__main__":
    main()
