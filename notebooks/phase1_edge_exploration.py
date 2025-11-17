"""
Ball Knower - Phase 1 Edge Exploration
======================================

Goal: Validate whether systematic Vegas errors exist in specific regimes
before building dedicated models.

Strategy:
1. Weather Edge - Do wind/temp/conditions create predictable total/spread errors?
2. Injury Edge - Does Vegas over/underreact to key injuries?
3. Referee Edge - Do certain crews systematically affect scoring?
4. Matchup Edge - Are extreme style matchups mispriced?

Decision criteria:
- Win rate > 52.4% (beat -110 vig)
- Sample size n ≥ 30 games
- Effect is explainable (not random noise)

If Phase 1 finds edges → proceed to Phase 2 (build models)
If Phase 1 finds nothing → pause or pivot
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from team_mapping import normalize_team_name

print("="*80)
print("Ball Knower - Phase 1: Edge Discovery")
print("="*80)

# ============================================================================
# 1. LOAD & PREP DATA
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: Load & Prepare Data")
print("="*80)

# Load schedules (game results, Vegas lines, weather)
schedules = pd.read_parquet('schedules.parquet')

# Filter to regular season games with complete data
df = schedules[
    (schedules['game_type'] == 'REG') &
    (schedules['season'] >= 2010) &  # Modern era
    (schedules['away_score'].notna()) &
    (schedules['home_score'].notna()) &
    (schedules['spread_line'].notna()) &
    (schedules['total_line'].notna())
].copy()

print(f"\nGames loaded: {len(df):,}")
print(f"Seasons: {df['season'].min()}-{df['season'].max()}")
print(f"Total games with data: {len(df):,}")

# Compute actual margins and totals
df['actual_margin'] = df['home_score'] - df['away_score']  # positive = home won by X
df['actual_total'] = df['home_score'] + df['away_score']

# Compute Vegas errors
# spread_line is from home team perspective (negative = home favored)
# If spread_line = -3, home is 3-point favorite
# If home wins by 7, actual_margin = 7, vegas_error_spread = 7 - (-3) = 10
# Positive error = home did better than expected
df['vegas_error_spread'] = df['actual_margin'] - df['spread_line']
df['vegas_error_total'] = df['actual_total'] - df['total_line']

# Betting outcomes
# ATS = Against The Spread (did home team cover?)
# Home covers if: actual_margin > spread_line
# (if home was -3 and won by 7, they covered: 7 > -3)
df['home_covered'] = (df['actual_margin'] > df['spread_line']).astype(int)
df['away_covered'] = (df['actual_margin'] < df['spread_line']).astype(int)
df['push_spread'] = (df['actual_margin'] == df['spread_line']).astype(int)

# Over/Under
df['over_hit'] = (df['actual_total'] > df['total_line']).astype(int)
df['under_hit'] = (df['actual_total'] < df['total_line']).astype(int)
df['push_total'] = (df['actual_total'] == df['total_line']).astype(int)

print(f"\n✓ Computed Vegas errors and betting outcomes")
print(f"  - Spread data: {df['spread_line'].notna().sum():,} games")
print(f"  - Total data: {df['total_line'].notna().sum():,} games")
print(f"  - Weather data: {df['wind'].notna().sum():,} games with wind")
print(f"  - Temperature data: {df['temp'].notna().sum():,} games with temp")

# ============================================================================
# 2. WEATHER EDGE EXPLORATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: Weather Edge Analysis")
print("="*80)

# Create weather bins
def categorize_wind(wind_mph):
    if pd.isna(wind_mph):
        return 'Unknown'
    elif wind_mph < 10:
        return '0-10 mph (calm)'
    elif wind_mph < 15:
        return '10-15 mph (moderate)'
    elif wind_mph < 20:
        return '15-20 mph (high)'
    else:
        return '20+ mph (extreme)'

def categorize_temp(temp_f):
    if pd.isna(temp_f):
        return 'Unknown'
    elif temp_f < 20:
        return '<20°F (extreme cold)'
    elif temp_f < 32:
        return '20-32°F (freezing)'
    elif temp_f < 50:
        return '32-50°F (cold)'
    elif temp_f < 70:
        return '50-70°F (moderate)'
    else:
        return '70°F+ (warm)'

def categorize_roof(roof):
    if pd.isna(roof):
        return 'Unknown'
    elif roof in ['dome', 'closed']:
        return 'Indoors'
    else:
        return 'Outdoors'

df['wind_category'] = df['wind'].apply(categorize_wind)
df['temp_category'] = df['temp'].apply(categorize_temp)
df['roof_category'] = df['roof'].apply(categorize_roof)

# ============================================================================
# 2A. Wind Analysis - Does wind affect totals?
# ============================================================================

print("\n" + "-"*80)
print("2A. Wind Impact on Totals")
print("-"*80)

wind_analysis = df[df['wind'].notna()].groupby('wind_category').agg({
    'game_id': 'count',
    'vegas_error_total': ['mean', 'std'],
    'under_hit': 'mean',
    'over_hit': 'mean',
    'push_total': 'mean'
}).round(3)

wind_analysis.columns = ['n_games', 'avg_total_error', 'std_total_error',
                         'under_rate', 'over_rate', 'push_rate']

# Order by wind severity
wind_order = ['0-10 mph (calm)', '10-15 mph (moderate)', '15-20 mph (high)', '20+ mph (extreme)']
wind_analysis = wind_analysis.reindex(wind_order)

print("\nWind Category Analysis:")
print(wind_analysis)

# Highlight high-wind games
high_wind_games = df[df['wind'] >= 15].copy()
if len(high_wind_games) >= 30:
    under_rate_high_wind = high_wind_games['under_hit'].mean()
    print(f"\n🔍 High Wind Games (≥15 mph):")
    print(f"   Sample size: {len(high_wind_games)} games")
    print(f"   Under hit rate: {under_rate_high_wind:.1%}")
    print(f"   Over hit rate: {high_wind_games['over_hit'].mean():.1%}")
    print(f"   Avg total error: {high_wind_games['vegas_error_total'].mean():.2f} points")

    # Does this beat the vig?
    # Need 52.4% to profit at -110
    if under_rate_high_wind >= 0.524:
        print(f"   ✓ POTENTIAL EDGE: Under rate {under_rate_high_wind:.1%} > 52.4% threshold")
        print(f"   → Betting unders in 15+ mph wind beats the vig!")
    else:
        print(f"   ✗ No edge: Under rate {under_rate_high_wind:.1%} < 52.4% threshold")
else:
    print(f"\n⚠️  Insufficient high-wind games (n={len(high_wind_games)})")

# ============================================================================
# 2B. Temperature Analysis - Does extreme cold affect scoring?
# ============================================================================

print("\n" + "-"*80)
print("2B. Temperature Impact on Totals")
print("-"*80)

temp_analysis = df[df['temp'].notna()].groupby('temp_category').agg({
    'game_id': 'count',
    'vegas_error_total': ['mean', 'std'],
    'under_hit': 'mean',
    'over_hit': 'mean',
}).round(3)

temp_analysis.columns = ['n_games', 'avg_total_error', 'std_total_error',
                         'under_rate', 'over_rate']

# Order by temperature
temp_order = ['<20°F (extreme cold)', '20-32°F (freezing)', '32-50°F (cold)',
              '50-70°F (moderate)', '70°F+ (warm)']
temp_analysis = temp_analysis.reindex(temp_order)

print("\nTemperature Category Analysis:")
print(temp_analysis)

# Extreme cold games
extreme_cold = df[df['temp'] < 32].copy()
if len(extreme_cold) >= 30:
    under_rate_cold = extreme_cold['under_hit'].mean()
    print(f"\n🔍 Extreme Cold Games (<32°F):")
    print(f"   Sample size: {len(extreme_cold)} games")
    print(f"   Under hit rate: {under_rate_cold:.1%}")
    print(f"   Over hit rate: {extreme_cold['over_hit'].mean():.1%}")
    print(f"   Avg total error: {extreme_cold['vegas_error_total'].mean():.2f} points")

    if under_rate_cold >= 0.524:
        print(f"   ✓ POTENTIAL EDGE: Under rate {under_rate_cold:.1%} > 52.4%")
    else:
        print(f"   ✗ No edge: Under rate {under_rate_cold:.1%} < 52.4%")
else:
    print(f"\n⚠️  Insufficient cold games (n={len(extreme_cold)})")

# ============================================================================
# 2C. Dome vs Outdoor - Control check
# ============================================================================

print("\n" + "-"*80)
print("2C. Indoor vs Outdoor (Control)")
print("-"*80)

roof_analysis = df.groupby('roof_category').agg({
    'game_id': 'count',
    'vegas_error_total': 'mean',
    'over_hit': 'mean',
    'under_hit': 'mean',
    'actual_total': 'mean',
    'total_line': 'mean'
}).round(2)

roof_analysis.columns = ['n_games', 'avg_total_error', 'over_rate',
                         'under_rate', 'avg_actual_total', 'avg_vegas_total']

print("\nIndoor vs Outdoor:")
print(roof_analysis)

# ============================================================================
# 3. INJURY EDGE EXPLORATION (Phase 1 - Simple)
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: Injury Impact Analysis (Simplified)")
print("="*80)

# For Phase 1, we'll do a simple QB-based analysis
# More sophisticated injury modeling comes in Phase 2 if this shows promise

# Load injury data
try:
    injuries = pd.read_parquet('injuries.parquet')
    print(f"\n✓ Loaded {len(injuries):,} injury records")
    print(f"Columns: {injuries.columns.tolist()}")

    # Quick peek at injury data structure
    print("\nSample injury record:")
    print(injuries.head(1).T)

    # We'll do deeper injury analysis in a separate section after understanding the schema
    print("\n⚠️  Injury analysis requires schema exploration - deferred to next iteration")

except Exception as e:
    print(f"\n⚠️  Could not load injury data: {e}")

# ============================================================================
# 4. REFEREE EDGE EXPLORATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: Referee Impact Analysis")
print("="*80)

# Analyze if certain refs systematically affect totals or spreads
ref_analysis = df[df['referee'].notna()].groupby('referee').agg({
    'game_id': 'count',
    'actual_total': 'mean',
    'total_line': 'mean',
    'vegas_error_total': 'mean',
    'over_hit': 'mean',
    'under_hit': 'mean'
}).round(2)

ref_analysis.columns = ['n_games', 'avg_actual_total', 'avg_vegas_total',
                        'avg_total_error', 'over_rate', 'under_rate']

# Filter to refs with enough games (min 30)
ref_analysis = ref_analysis[ref_analysis['n_games'] >= 30].copy()
ref_analysis = ref_analysis.sort_values('avg_total_error', ascending=False)

print(f"\nReferees with ≥30 games: {len(ref_analysis)}")
print("\nTop 10 refs (games go OVER Vegas total most):")
print(ref_analysis.head(10))

print("\nBottom 10 refs (games go UNDER Vegas total most):")
print(ref_analysis.tail(10))

# Check if any ref has a systematic edge
extreme_refs = ref_analysis[
    (ref_analysis['n_games'] >= 50) &
    ((ref_analysis['over_rate'] >= 0.55) | (ref_analysis['under_rate'] >= 0.55))
]

if len(extreme_refs) > 0:
    print(f"\n🔍 Referees with 55%+ over/under rate (n≥50):")
    print(extreme_refs)
    print("\n   → Potential edge in betting against ref tendencies")
else:
    print(f"\n✗ No referees show systematic 55%+ edge")

# ============================================================================
# 5. SUMMARY & PHASE 1 DECISION
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: Phase 1 Summary & Go/No-Go Decision")
print("="*80)

print("\n📊 Phase 1 Exploration Results:")
print("-" * 80)

# Weather findings
print("\n1. WEATHER EDGE:")
if len(high_wind_games) >= 30:
    under_rate_wind = high_wind_games['under_hit'].mean()
    if under_rate_wind >= 0.524:
        print(f"   ✓ HIGH WIND (≥15 mph): Under rate {under_rate_wind:.1%}")
        print(f"     Sample: {len(high_wind_games)} games")
        print(f"     → PROCEED to Phase 2: Build wind-based total model")
    else:
        print(f"   ✗ High wind: {under_rate_wind:.1%} under rate (below 52.4%)")
else:
    print(f"   ⚠️  Insufficient data (n={len(high_wind_games)})")

if len(extreme_cold) >= 30:
    under_rate_temp = extreme_cold['under_hit'].mean()
    if under_rate_temp >= 0.524:
        print(f"   ✓ EXTREME COLD (<32°F): Under rate {under_rate_temp:.1%}")
        print(f"     Sample: {len(extreme_cold)} games")
        print(f"     → PROCEED to Phase 2: Build temperature-based total model")
    else:
        print(f"   ✗ Extreme cold: {under_rate_temp:.1%} under rate (below 52.4%)")
else:
    print(f"   ⚠️  Insufficient cold weather data")

# Referee findings
print("\n2. REFEREE EDGE:")
if len(extreme_refs) > 0:
    print(f"   ✓ Found {len(extreme_refs)} refs with 55%+ edge (n≥50)")
    print(f"     → PROCEED to Phase 2: Build referee-based model")
else:
    print(f"   ✗ No referees with systematic edge found")

# Injury findings
print("\n3. INJURY EDGE:")
print(f"   ⚠️  Requires deeper analysis (Phase 1b)")
print(f"   → Next step: Analyze QB injuries, WR1 injuries, etc.")

print("\n" + "="*80)
print("PHASE 1 DECISION FRAMEWORK:")
print("="*80)
print("""
✓ PROCEED to Phase 2 if:
  - At least 1 edge shows >52.4% win rate with n≥30
  - Effect is explainable (not random noise)
  - Can build predictive model for future games

✗ PAUSE if:
  - No edges found in any category
  - Edges exist but sample size too small (n<30)
  - Edges are not actionable (can't predict future games)

→ CURRENT STATUS: Running analysis...
""")

# Count validated edges
edges_found = []

if len(high_wind_games) >= 30 and high_wind_games['under_hit'].mean() >= 0.524:
    edges_found.append("High Wind Unders")

if len(extreme_cold) >= 30 and extreme_cold['under_hit'].mean() >= 0.524:
    edges_found.append("Cold Weather Unders")

if len(extreme_refs) > 0:
    edges_found.append("Referee Tendencies")

print(f"\n📈 EDGES VALIDATED: {len(edges_found)}")
if edges_found:
    for edge in edges_found:
        print(f"   ✓ {edge}")
    print(f"\n🚀 RECOMMENDATION: PROCEED TO PHASE 2")
    print(f"   Build specialized models for validated edges")
else:
    print(f"   (None found yet - injury analysis pending)")
    print(f"\n⏸️  RECOMMENDATION: Complete injury analysis before decision")

print("\n" + "="*80)
print("Analysis complete. Review findings above.")
print("="*80)
