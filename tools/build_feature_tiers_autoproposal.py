#!/usr/bin/env python3
"""
Ball_Knower v2.0 - Feature Tier Auto-Proposal Tool

This standalone script reads the feature catalog and automatically assigns
each column to a tier based on its role and leakage risk.

Usage:
    python tools/build_feature_tiers_autoproposal.py

Inputs:
    - data/_feature_catalog_raw.csv

Outputs:
    - data/_feature_tiers_autoproposed.csv (machine-readable)
    - docs/FEATURE_TIERS_AUTOPROPOSAL.md (human-readable report)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import csv
from collections import defaultdict, Counter

# ============================================================================
# Configuration
# ============================================================================

INPUT_CATALOG = Path("data/_feature_catalog_raw.csv")
OUTPUT_CSV = Path("data/_feature_tiers_autoproposed.csv")
OUTPUT_MD = Path("docs/FEATURE_TIERS_AUTOPROPOSAL.md")

# Tier definitions
TIER_KEYS_STRUCTURE = "T0_KEYS_STRUCTURE"
TIER_TEAM_STRENGTH = "T1_TEAM_STRENGTH"
TIER_MARKET = "T2_MARKET"
TIER_EXPERIMENTAL = "T3_EXPERIMENTAL"
TIER_FORBIDDEN = "TX_FORBIDDEN"

ALL_TIERS = [
    TIER_KEYS_STRUCTURE,
    TIER_TEAM_STRENGTH,
    TIER_MARKET,
    TIER_EXPERIMENTAL,
    TIER_FORBIDDEN,
]


# ============================================================================
# Tiering Logic
# ============================================================================

def assign_tier(row: Dict[str, str]) -> Tuple[str, str]:
    """
    Assign a tier to a column based on its role and leakage risk.

    Returns:
        (tier_name, reason)
    """
    column_name = row.get("column_name", "").strip().lower()
    role = row.get("role", "").strip().lower()
    leakage = row.get("leakage_risk", "").strip().lower()

    # Normalize empty/missing values
    if not role or role == "unknown":
        role = "unknown"
    if not leakage or leakage == "unknown":
        leakage = "unknown"

    # ========================================================================
    # Tier X – Forbidden (check first)
    # ========================================================================

    # High leakage risk = automatic ban
    if leakage == "high":
        return TIER_FORBIDDEN, "High leakage risk – never use in pre-game models"

    # Explicit outcome columns
    outcome_keywords = [
        "result", "margin", "winner", "loser", "outcome", "actual",
        "final_score", "pts_scored", "pts_allowed", "total_points",
        "ats_outcome", "totals_outcome", "covered", "under", "over"
    ]
    if any(kw in column_name for kw in outcome_keywords):
        return TIER_FORBIDDEN, f"Column name '{column_name}' suggests post-game outcome"

    # In-game stats are forbidden for pre-game models
    if role == "in_game_stats":
        return TIER_FORBIDDEN, "In-game stats – not available before game starts"

    # Post-game role
    if role == "post_game" or "post_game" in role:
        return TIER_FORBIDDEN, "Post-game data – not available for pre-game modeling"

    # ========================================================================
    # Tier 0 – Keys & Structure
    # ========================================================================

    if role in ["id_key", "pre_game_structure"]:
        if leakage in ["low", "unknown"]:
            return TIER_KEYS_STRUCTURE, f"Role: {role} – structural/identifier column"

    # Common structural columns by name
    structural_names = [
        "game_id", "season", "week", "home_team", "away_team", "team",
        "stadium", "surface", "roof", "is_playoff", "is_neutral",
        "game_date", "kickoff_time", "day_of_week"
    ]
    if column_name in structural_names and leakage in ["low", "unknown"]:
        return TIER_KEYS_STRUCTURE, f"Structural identifier: {column_name}"

    # ========================================================================
    # Tier 1 – Pre-Game Team Strength
    # ========================================================================

    if role == "pre_game_team_strength":
        if leakage in ["low", "unknown"]:
            return TIER_TEAM_STRENGTH, "Pre-game team strength metric (Elo, ratings, etc.)"

    # Strength-related columns by name pattern
    strength_keywords = ["elo", "nfelo", "rating", "power", "qbr", "strength", "rank"]
    if any(kw in column_name for kw in strength_keywords):
        if leakage in ["low", "unknown"]:
            return TIER_TEAM_STRENGTH, f"Team strength indicator: {column_name}"

    # ========================================================================
    # Tier 2 – Pre-Game Market
    # ========================================================================

    if role == "pre_game_market":
        if leakage in ["low", "unknown"]:
            return TIER_MARKET, "Pre-game market data (spreads, totals, probabilities)"

    # Market-related columns by name
    market_keywords = [
        "spread", "line", "total", "over_under", "ou", "moneyline", "ml",
        "implied_prob", "opening", "closing", "consensus", "odds"
    ]
    if any(kw in column_name for kw in market_keywords):
        if leakage in ["low", "unknown"]:
            return TIER_MARKET, f"Market/betting data: {column_name}"

    # ========================================================================
    # Tier 3 – Experimental / Ambiguous
    # ========================================================================

    # Medium leakage or unclear role → experimental
    if leakage == "medium":
        return TIER_EXPERIMENTAL, "Medium leakage risk – requires manual review"

    if role in ["meta_misc", "derived", "experimental"]:
        return TIER_EXPERIMENTAL, f"Role: {role} – needs human judgment"

    # Unknown role/leakage but not clearly forbidden
    if role == "unknown" and leakage in ["low", "unknown"]:
        return TIER_EXPERIMENTAL, "Unknown role – classify manually after review"

    # ========================================================================
    # Default: Experimental
    # ========================================================================

    return TIER_EXPERIMENTAL, f"Unclear classification (role={role}, leakage={leakage})"


# ============================================================================
# Processing
# ============================================================================

def load_catalog(catalog_path: Path) -> List[Dict[str, str]]:
    """Load the feature catalog CSV."""
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"\n❌ Feature catalog not found: {catalog_path}\n\n"
            f"Please run the feature catalog generation tool first to create:\n"
            f"  {catalog_path}\n"
        )

    with open(catalog_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Feature catalog is empty: {catalog_path}")

    return rows


def assign_all_tiers(catalog: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Assign tiers to all columns in the catalog."""
    results = []

    for row in catalog:
        tier, reason = assign_tier(row)

        # Build output row with tier assignment
        output_row = {
            "column_name": row.get("column_name", ""),
            "role": row.get("role", ""),
            "leakage_risk": row.get("leakage_risk", ""),
            "tier": tier,
            "reason": reason,
            "source_files": row.get("source_files", ""),
            "missing_rate_overall": row.get("missing_rate_overall", ""),
            "dtype_candidates": row.get("dtype_candidates", ""),
            "example_values": row.get("example_values", ""),
            "notes": row.get("notes", ""),
        }

        results.append(output_row)

    return results


def write_csv_output(data: List[Dict[str, str]], output_path: Path):
    """Write the tier assignments to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "column_name",
        "role",
        "leakage_risk",
        "tier",
        "reason",
        "source_files",
        "missing_rate_overall",
        "dtype_candidates",
        "example_values",
        "notes",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def generate_markdown_report(data: List[Dict[str, str]], output_path: Path):
    """Generate a human-readable Markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by tier
    tier_groups = defaultdict(list)
    for row in data:
        tier_groups[row["tier"]].append(row)

    # Count stats
    tier_counts = Counter(row["tier"] for row in data)
    leakage_by_tier = defaultdict(Counter)
    for row in data:
        leakage_by_tier[row["tier"]][row["leakage_risk"]] += 1

    # Build markdown
    lines = []
    lines.append("# Ball_Knower v2.0 – Feature Tier Auto-Proposal Report\n")
    lines.append("**Auto-generated by:** `tools/build_feature_tiers_autoproposal.py`\n")
    lines.append("**Input:** `data/_feature_catalog_raw.csv`\n")
    lines.append("**Output:** `data/_feature_tiers_autoproposed.csv`\n")
    lines.append("---\n")

    # Overview
    lines.append("## Overview\n")
    lines.append(f"**Total Columns Analyzed:** {len(data)}\n")
    lines.append("### Tier Distribution\n")
    for tier in ALL_TIERS:
        count = tier_counts[tier]
        pct = (count / len(data) * 100) if data else 0
        lines.append(f"- **{tier}**: {count} columns ({pct:.1f}%)")
    lines.append("")

    lines.append("### Leakage Risk by Tier\n")
    for tier in ALL_TIERS:
        if tier_counts[tier] == 0:
            continue
        leakage_dist = leakage_by_tier[tier]
        leakage_str = ", ".join(f"{k}={v}" for k, v in sorted(leakage_dist.items()))
        lines.append(f"- **{tier}**: {leakage_str}")
    lines.append("\n---\n")

    # Tier 0
    lines.append("## Tier 0 – Keys & Structure\n")
    lines.append("**Purpose:** Identifier and structural columns needed for data organization.\n")
    lines.append("**Usage:** Always allowed in all models (non-predictive).\n")
    if tier_counts[TIER_KEYS_STRUCTURE] > 0:
        lines.append(f"\n**Count:** {tier_counts[TIER_KEYS_STRUCTURE]} columns\n")
        lines.append("| Column Name | Role | Leakage | Reason |")
        lines.append("|-------------|------|---------|--------|")
        for row in tier_groups[TIER_KEYS_STRUCTURE]:
            lines.append(
                f"| {row['column_name']} | {row['role']} | {row['leakage_risk']} | {row['reason']} |"
            )
    else:
        lines.append("\n*No columns assigned to this tier.*\n")
    lines.append("\n---\n")

    # Tier 1
    lines.append("## Tier 1 – Pre-Game Team Strength\n")
    lines.append("**Purpose:** Long-term team strength indicators (Elo, power ratings, QBR, etc.).\n")
    lines.append("**Usage:** Core predictors for v2.0 base model.\n")
    if tier_counts[TIER_TEAM_STRENGTH] > 0:
        lines.append(f"\n**Count:** {tier_counts[TIER_TEAM_STRENGTH]} columns\n")
        lines.append("| Column Name | Role | Leakage | Reason |")
        lines.append("|-------------|------|---------|--------|")
        for row in tier_groups[TIER_TEAM_STRENGTH]:
            lines.append(
                f"| {row['column_name']} | {row['role']} | {row['leakage_risk']} | {row['reason']} |"
            )
    else:
        lines.append("\n*No columns assigned to this tier.*\n")
    lines.append("\n---\n")

    # Tier 2
    lines.append("## Tier 2 – Pre-Game Market\n")
    lines.append("**Purpose:** Market-derived features (spreads, totals, implied probabilities).\n")
    lines.append("**Usage:** Can be used in v2.0 models; consider as separate feature set for A/B testing.\n")
    if tier_counts[TIER_MARKET] > 0:
        lines.append(f"\n**Count:** {tier_counts[TIER_MARKET]} columns\n")
        lines.append("| Column Name | Role | Leakage | Reason |")
        lines.append("|-------------|------|---------|--------|")
        for row in tier_groups[TIER_MARKET]:
            lines.append(
                f"| {row['column_name']} | {row['role']} | {row['leakage_risk']} | {row['reason']} |"
            )
    else:
        lines.append("\n*No columns assigned to this tier.*\n")
    lines.append("\n---\n")

    # Tier 3
    lines.append("## Tier 3 – Experimental / Ambiguous\n")
    lines.append("**Purpose:** Columns with unclear role or medium leakage risk.\n")
    lines.append("**Usage:** Requires manual human review before use in any model.\n")
    if tier_counts[TIER_EXPERIMENTAL] > 0:
        lines.append(f"\n**Count:** {tier_counts[TIER_EXPERIMENTAL]} columns\n")
        lines.append("| Column Name | Role | Leakage | Reason |")
        lines.append("|-------------|------|---------|--------|")
        for row in tier_groups[TIER_EXPERIMENTAL]:
            lines.append(
                f"| {row['column_name']} | {row['role']} | {row['leakage_risk']} | {row['reason']} |"
            )
    else:
        lines.append("\n*No columns assigned to this tier.*\n")
    lines.append("\n---\n")

    # Tier X
    lines.append("## Tier X – Forbidden\n")
    lines.append("**Purpose:** Post-game outcomes and high-leakage columns.\n")
    lines.append("**Usage:** ❌ **NEVER USE IN PRE-GAME MODELS** – These columns leak the answer.\n")
    if tier_counts[TIER_FORBIDDEN] > 0:
        lines.append(f"\n**Count:** {tier_counts[TIER_FORBIDDEN]} columns\n")
        lines.append("| Column Name | Role | Leakage | Reason |")
        lines.append("|-------------|------|---------|--------|")
        for row in tier_groups[TIER_FORBIDDEN]:
            lines.append(
                f"| {row['column_name']} | {row['role']} | {row['leakage_risk']} | {row['reason']} |"
            )
    else:
        lines.append("\n*No columns assigned to this tier.*\n")
    lines.append("\n---\n")

    # Recommendations
    lines.append("## Recommendations\n")
    lines.append("### ✅ Safe to Use (v2.0 Base Model)\n")
    lines.append(f"- **Tier 0** ({tier_counts[TIER_KEYS_STRUCTURE]} cols): Always include for structure")
    lines.append(f"- **Tier 1** ({tier_counts[TIER_TEAM_STRENGTH]} cols): Core team strength predictors")
    lines.append(f"- **Tier 2** ({tier_counts[TIER_MARKET]} cols): Market features (optional, test separately)")
    lines.append("")

    lines.append("### ⚠️ Requires Manual Review\n")
    lines.append(f"- **Tier 3** ({tier_counts[TIER_EXPERIMENTAL]} cols): Review each column individually")
    lines.append("  - Check definitions and example values")
    lines.append("  - Reclassify to appropriate tier or mark as forbidden")
    lines.append("")

    lines.append("### ❌ Exclude from ALL Training\n")
    lines.append(f"- **Tier X** ({tier_counts[TIER_FORBIDDEN]} cols): Post-game outcomes and high-leakage data")
    lines.append("  - Do not use these columns in any pre-game model")
    lines.append("  - Keep for post-game analysis and validation only")
    lines.append("")

    lines.append("---\n")
    lines.append("## Next Steps\n")
    lines.append("1. **Review Tier 3 columns** – Manually classify ambiguous features")
    lines.append("2. **Validate Tier 1 columns** – Confirm all team strength features are pre-game only")
    lines.append("3. **Test Tier 2 separately** – Run baseline model with/without market features")
    lines.append("4. **Audit Tier X** – Double-check no post-game leakage in other tiers")
    lines.append("5. **Update FEATURE_TIERS.md** – Create final authoritative tier map after review")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_console_summary(data: List[Dict[str, str]]):
    """Print a concise summary to console."""
    tier_counts = Counter(row["tier"] for row in data)
    leakage_by_tier = defaultdict(Counter)
    for row in data:
        leakage_by_tier[row["tier"]][row["leakage_risk"]] += 1

    print("\n" + "=" * 70)
    print("Feature Tier Auto-Proposal Summary")
    print("=" * 70)
    print(f"\nTotal columns analyzed: {len(data)}\n")

    print("Tier Distribution:")
    print("-" * 70)
    for tier in ALL_TIERS:
        count = tier_counts[tier]
        pct = (count / len(data) * 100) if data else 0
        print(f"  {tier:25s} {count:4d} columns ({pct:5.1f}%)")

    print("\nLeakage Risk by Tier:")
    print("-" * 70)
    for tier in ALL_TIERS:
        if tier_counts[tier] == 0:
            continue
        leakage_dist = leakage_by_tier[tier]
        leakage_str = ", ".join(f"{k}={v}" for k, v in sorted(leakage_dist.items()))
        print(f"  {tier:25s} {leakage_str}")

    print("\n" + "=" * 70)
    print("✅ Outputs written:")
    print(f"   - {OUTPUT_CSV}")
    print(f"   - {OUTPUT_MD}")
    print("=" * 70 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution function."""
    print("\n🏈 Ball_Knower v2.0 – Feature Tier Auto-Proposal Tool\n")

    try:
        # Load catalog
        print(f"📖 Loading feature catalog: {INPUT_CATALOG}")
        catalog = load_catalog(INPUT_CATALOG)
        print(f"   ✓ Loaded {len(catalog)} columns\n")

        # Assign tiers
        print("🔍 Assigning tiers based on role and leakage risk...")
        results = assign_all_tiers(catalog)
        print(f"   ✓ Assigned tiers to {len(results)} columns\n")

        # Write CSV output
        print(f"💾 Writing machine-readable CSV: {OUTPUT_CSV}")
        write_csv_output(results, OUTPUT_CSV)
        print(f"   ✓ CSV written\n")

        # Write Markdown report
        print(f"📝 Generating Markdown report: {OUTPUT_MD}")
        generate_markdown_report(results, OUTPUT_MD)
        print(f"   ✓ Markdown report written\n")

        # Print summary
        print_console_summary(results)

        print("✅ Feature tier auto-proposal complete!\n")
        return 0

    except FileNotFoundError as e:
        print(f"\n{e}\n", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
