"""
Phase 2 Model Backtest - Out-of-Sample Validation

Test edge detection models on 2024 season (data not used in Phase 1 model design)

Validates:
1. WindTotalModel - Do high wind games still go under?
2. RefereeTotalModel - Do ref tendencies persist?
3. EnsembleModel - Does multi-signal confirmation improve performance?

Success Criteria:
- Win rate ≥ 52% (beat the vig)
- ROI > 0%
- Each model profitable independently
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from edge_models import (
    WindTotalModel,
    RefereeTotalModel,
    EnsembleModel,
    BetType,
    create_default_ensemble
)

print("="*80)
print("Phase 2: Edge Model Backtest (2024 Out-of-Sample Validation)")
print("="*80)

# ============================================================================
# 1. LOAD 2024 SEASON DATA
# ============================================================================

print("\nLoading 2024 season data...")

schedules = pd.read_parquet('schedules.parquet')

# Filter to 2024 regular season games only
df_2024 = schedules[
    (schedules['season'] == 2024) &
    (schedules['game_type'] == 'REG') &
    (schedules['away_score'].notna()) &  # Only completed games
    (schedules['home_score'].notna())
].copy()

print(f"✓ Loaded {len(df_2024)} completed 2024 games")
print(f"  Weeks: {df_2024['week'].min()}-{df_2024['week'].max()}")

# Compute actual outcomes
df_2024['actual_total'] = df_2024['home_score'] + df_2024['away_score']
df_2024['total_error'] = df_2024['actual_total'] - df_2024['total_line']

# Over/Under outcomes
df_2024['over_hit'] = (df_2024['actual_total'] > df_2024['total_line']).astype(int)
df_2024['under_hit'] = (df_2024['actual_total'] < df_2024['total_line']).astype(int)
df_2024['push_total'] = (df_2024['actual_total'] == df_2024['total_line']).astype(int)

print(f"\n2024 Season Summary:")
print(f"  Over hit rate: {df_2024['over_hit'].mean():.1%}")
print(f"  Under hit rate: {df_2024['under_hit'].mean():.1%}")
print(f"  Push rate: {df_2024['push_total'].mean():.1%}")

# ============================================================================
# 2. INITIALIZE MODELS
# ============================================================================

print("\n" + "="*80)
print("Initializing Edge Models")
print("="*80)

# Individual models
wind_model = WindTotalModel(wind_threshold=15.0, require_outdoor=True)
ref_model = RefereeTotalModel(min_games=50, only_active=True)

# Ensemble models
ensemble_conservative = create_default_ensemble(mode="CONSERVATIVE")
ensemble_aggressive = create_default_ensemble(mode="AGGRESSIVE")

print("\n✓ Models initialized:")
print(f"  1. WindTotalModel (≥15 mph outdoor)")
print(f"  2. RefereeTotalModel (active refs only, min 50 games)")
print(f"  3. EnsembleModel (Conservative: ≥2 models)")
print(f"  4. EnsembleModel (Aggressive: any model)")

# ============================================================================
# 3. RUN BACKTEST
# ============================================================================

print("\n" + "="*80)
print("Running Backtest on 2024 Season")
print("="*80)

results = []

for idx, row in df_2024.iterrows():
    # Convert row to dict for model input
    game = row.to_dict()

    # Run individual models
    wind_pred = wind_model.predict(game)
    ref_pred = ref_model.predict(game)

    # Run ensembles
    ensemble_cons = ensemble_conservative.predict(game)
    ensemble_agg = ensemble_aggressive.predict(game)

    # Store results
    result = {
        'game_id': game['game_id'],
        'week': game['week'],
        'away_team': game['away_team'],
        'home_team': game['home_team'],
        'wind': game.get('wind', np.nan),
        'referee': game.get('referee', 'Unknown'),
        'total_line': game['total_line'],
        'actual_total': game['actual_total'],
        'total_error': game['total_error'],
        'over_hit': game['over_hit'],
        'under_hit': game['under_hit'],

        # Wind model
        'wind_bet': wind_pred.bet_type.value,
        'wind_confidence': wind_pred.confidence,
        'wind_reasoning': wind_pred.reasoning,

        # Ref model
        'ref_bet': ref_pred.bet_type.value,
        'ref_confidence': ref_pred.confidence,
        'ref_reasoning': ref_pred.reasoning,

        # Ensemble (conservative)
        'ens_cons_bet': ensemble_cons['total_bet'].value,
        'ens_cons_confidence': ensemble_cons['confidence'],
        'ens_cons_models': ','.join(ensemble_cons['models_agreeing']),

        # Ensemble (aggressive)
        'ens_agg_bet': ensemble_agg['total_bet'].value,
        'ens_agg_confidence': ensemble_agg['confidence'],
        'ens_agg_models': ','.join(ensemble_agg['models_agreeing']),
    }

    results.append(result)

results_df = pd.DataFrame(results)

print(f"\n✓ Backtest complete: {len(results_df)} games analyzed")

# ============================================================================
# 4. EVALUATE WIND MODEL
# ============================================================================

print("\n" + "="*80)
print("Wind Model Performance")
print("="*80)

wind_bets = results_df[results_df['wind_bet'] != 'NO_BET'].copy()

if len(wind_bets) > 0:
    print(f"\nBets placed: {len(wind_bets)}")

    # Calculate wins
    wind_bets['win'] = 0
    wind_bets.loc[
        (wind_bets['wind_bet'] == 'UNDER') & (wind_bets['under_hit'] == 1),
        'win'
    ] = 1
    wind_bets.loc[
        (wind_bets['wind_bet'] == 'OVER') & (wind_bets['over_hit'] == 1),
        'win'
    ] = 1

    win_rate = wind_bets['win'].mean()
    wins = wind_bets['win'].sum()
    losses = len(wind_bets) - wins

    # ROI calculation (assuming -110 vig)
    # Win 1 unit per win, lose 1.1 units per loss
    profit = wins * 1.0 - losses * 1.1
    roi = (profit / len(wind_bets)) * 100 if len(wind_bets) > 0 else 0

    print(f"\nResults:")
    print(f"  Win rate: {win_rate:.1%} ({wins}W - {losses}L)")
    print(f"  ROI: {roi:+.1f}%")
    print(f"  Profit: {profit:+.1f} units")

    # Break down by bet type
    under_bets = wind_bets[wind_bets['wind_bet'] == 'UNDER']
    if len(under_bets) > 0:
        under_win_rate = under_bets['win'].mean()
        print(f"\n  Under bets: {len(under_bets)} ({under_win_rate:.1%} win rate)")

    # Show sample bets
    print(f"\nSample bets:")
    sample = wind_bets[['week', 'away_team', 'home_team', 'wind', 'wind_bet',
                        'total_line', 'actual_total', 'win']].head(10)
    print(sample.to_string(index=False))

    # Validation
    if win_rate >= 0.524:
        print(f"\n✅ PASS: Win rate {win_rate:.1%} ≥ 52.4% threshold")
    else:
        print(f"\n❌ FAIL: Win rate {win_rate:.1%} < 52.4% threshold")

else:
    print("\n⚠️  No bets placed by wind model in 2024")

# ============================================================================
# 5. EVALUATE REFEREE MODEL
# ============================================================================

print("\n" + "="*80)
print("Referee Model Performance")
print("="*80)

ref_bets = results_df[results_df['ref_bet'] != 'NO_BET'].copy()

if len(ref_bets) > 0:
    print(f"\nBets placed: {len(ref_bets)}")

    # Calculate wins
    ref_bets['win'] = 0
    ref_bets.loc[
        (ref_bets['ref_bet'] == 'UNDER') & (ref_bets['under_hit'] == 1),
        'win'
    ] = 1
    ref_bets.loc[
        (ref_bets['ref_bet'] == 'OVER') & (ref_bets['over_hit'] == 1),
        'win'
    ] = 1

    win_rate = ref_bets['win'].mean()
    wins = ref_bets['win'].sum()
    losses = len(ref_bets) - wins

    # ROI
    profit = wins * 1.0 - losses * 1.1
    roi = (profit / len(ref_bets)) * 100 if len(ref_bets) > 0 else 0

    print(f"\nResults:")
    print(f"  Win rate: {win_rate:.1%} ({wins}W - {losses}L)")
    print(f"  ROI: {roi:+.1f}%")
    print(f"  Profit: {profit:+.1f} units")

    # Break down by referee
    print(f"\nBy referee:")
    ref_summary = ref_bets.groupby('referee').agg({
        'win': ['count', 'sum', 'mean']
    }).round(3)
    ref_summary.columns = ['bets', 'wins', 'win_rate']
    print(ref_summary)

    # Show sample bets
    print(f"\nSample bets:")
    sample = ref_bets[['week', 'away_team', 'home_team', 'referee', 'ref_bet',
                       'total_line', 'actual_total', 'win']].head(10)
    print(sample.to_string(index=False))

    # Validation
    if win_rate >= 0.524:
        print(f"\n✅ PASS: Win rate {win_rate:.1%} ≥ 52.4% threshold")
    else:
        print(f"\n❌ FAIL: Win rate {win_rate:.1%} < 52.4% threshold")

else:
    print("\n⚠️  No bets placed by referee model in 2024")

# ============================================================================
# 6. EVALUATE ENSEMBLE (CONSERVATIVE)
# ============================================================================

print("\n" + "="*80)
print("Ensemble Model Performance (Conservative: ≥2 models)")
print("="*80)

ens_cons_bets = results_df[results_df['ens_cons_bet'] != 'NO_BET'].copy()

if len(ens_cons_bets) > 0:
    print(f"\nBets placed: {len(ens_cons_bets)}")

    # Calculate wins
    ens_cons_bets['win'] = 0
    ens_cons_bets.loc[
        (ens_cons_bets['ens_cons_bet'] == 'UNDER') & (ens_cons_bets['under_hit'] == 1),
        'win'
    ] = 1
    ens_cons_bets.loc[
        (ens_cons_bets['ens_cons_bet'] == 'OVER') & (ens_cons_bets['over_hit'] == 1),
        'win'
    ] = 1

    win_rate = ens_cons_bets['win'].mean()
    wins = ens_cons_bets['win'].sum()
    losses = len(ens_cons_bets) - wins

    # ROI
    profit = wins * 1.0 - losses * 1.1
    roi = (profit / len(ens_cons_bets)) * 100 if len(ens_cons_bets) > 0 else 0

    print(f"\nResults:")
    print(f"  Win rate: {win_rate:.1%} ({wins}W - {losses}L)")
    print(f"  ROI: {roi:+.1f}%")
    print(f"  Profit: {profit:+.1f} units")

    # Show which models agreed
    print(f"\nModels agreeing:")
    print(ens_cons_bets['ens_cons_models'].value_counts())

    # Show sample bets
    print(f"\nSample bets:")
    sample = ens_cons_bets[['week', 'away_team', 'home_team', 'ens_cons_bet',
                            'ens_cons_models', 'total_line', 'actual_total', 'win']].head(10)
    print(sample.to_string(index=False))

    # Validation
    if win_rate >= 0.524:
        print(f"\n✅ PASS: Win rate {win_rate:.1%} ≥ 52.4% threshold")
    else:
        print(f"\n❌ FAIL: Win rate {win_rate:.1%} < 52.4% threshold")

else:
    print("\n⚠️  No bets placed by conservative ensemble in 2024")
    print("  (Requires ≥2 models to agree)")

# ============================================================================
# 7. EVALUATE ENSEMBLE (AGGRESSIVE)
# ============================================================================

print("\n" + "="*80)
print("Ensemble Model Performance (Aggressive: Any Model)")
print("="*80)

ens_agg_bets = results_df[results_df['ens_agg_bet'] != 'NO_BET'].copy()

if len(ens_agg_bets) > 0:
    print(f"\nBets placed: {len(ens_agg_bets)}")

    # Calculate wins
    ens_agg_bets['win'] = 0
    ens_agg_bets.loc[
        (ens_agg_bets['ens_agg_bet'] == 'UNDER') & (ens_agg_bets['under_hit'] == 1),
        'win'
    ] = 1
    ens_agg_bets.loc[
        (ens_agg_bets['ens_agg_bet'] == 'OVER') & (ens_agg_bets['over_hit'] == 1),
        'win'
    ] = 1

    win_rate = ens_agg_bets['win'].mean()
    wins = ens_agg_bets['win'].sum()
    losses = len(ens_agg_bets) - wins

    # ROI
    profit = wins * 1.0 - losses * 1.1
    roi = (profit / len(ens_agg_bets)) * 100 if len(ens_agg_bets) > 0 else 0

    print(f"\nResults:")
    print(f"  Win rate: {win_rate:.1%} ({wins}W - {losses}L)")
    print(f"  ROI: {roi:+.1f}%")
    print(f"  Profit: {profit:+.1f} units")

    # Validation
    if win_rate >= 0.524:
        print(f"\n✅ PASS: Win rate {win_rate:.1%} ≥ 52.4% threshold")
    else:
        print(f"\n❌ FAIL: Win rate {win_rate:.1%} < 52.4% threshold")

else:
    print("\n⚠️  No bets placed by aggressive ensemble in 2024")

# ============================================================================
# 8. SUMMARY & GO/NO-GO DECISION
# ============================================================================

print("\n" + "="*80)
print("Phase 2 Backtest Summary")
print("="*80)

print(f"\n2024 Season (Out-of-Sample Test):")
print(f"  Total games: {len(df_2024)}")

# Compile results
summary_data = []

if len(wind_bets) > 0:
    wind_win_rate = wind_bets['win'].mean()
    wind_profit = wind_bets['win'].sum() * 1.0 - (len(wind_bets) - wind_bets['win'].sum()) * 1.1
    wind_roi = (wind_profit / len(wind_bets)) * 100
    summary_data.append({
        'Model': 'WindTotal',
        'Bets': len(wind_bets),
        'Win Rate': f"{wind_win_rate:.1%}",
        'ROI': f"{wind_roi:+.1f}%",
        'Profit': f"{wind_profit:+.1f}u",
        'Status': '✅ PASS' if wind_win_rate >= 0.524 else '❌ FAIL'
    })

if len(ref_bets) > 0:
    ref_win_rate = ref_bets['win'].mean()
    ref_profit = ref_bets['win'].sum() * 1.0 - (len(ref_bets) - ref_bets['win'].sum()) * 1.1
    ref_roi = (ref_profit / len(ref_bets)) * 100
    summary_data.append({
        'Model': 'RefereeTotal',
        'Bets': len(ref_bets),
        'Win Rate': f"{ref_win_rate:.1%}",
        'ROI': f"{ref_roi:+.1f}%",
        'Profit': f"{ref_profit:+.1f}u",
        'Status': '✅ PASS' if ref_win_rate >= 0.524 else '❌ FAIL'
    })

if len(ens_cons_bets) > 0:
    ens_cons_win_rate = ens_cons_bets['win'].mean()
    ens_cons_profit = ens_cons_bets['win'].sum() * 1.0 - (len(ens_cons_bets) - ens_cons_bets['win'].sum()) * 1.1
    ens_cons_roi = (ens_cons_profit / len(ens_cons_bets)) * 100
    summary_data.append({
        'Model': 'Ensemble (Conservative)',
        'Bets': len(ens_cons_bets),
        'Win Rate': f"{ens_cons_win_rate:.1%}",
        'ROI': f"{ens_cons_roi:+.1f}%",
        'Profit': f"{ens_cons_profit:+.1f}u",
        'Status': '✅ PASS' if ens_cons_win_rate >= 0.524 else '❌ FAIL'
    })

if len(ens_agg_bets) > 0:
    ens_agg_win_rate = ens_agg_bets['win'].mean()
    ens_agg_profit = ens_agg_bets['win'].sum() * 1.0 - (len(ens_agg_bets) - ens_agg_bets['win'].sum()) * 1.1
    ens_agg_roi = (ens_agg_profit / len(ens_agg_bets)) * 100
    summary_data.append({
        'Model': 'Ensemble (Aggressive)',
        'Bets': len(ens_agg_bets),
        'Win Rate': f"{ens_agg_win_rate:.1%}",
        'ROI': f"{ens_agg_roi:+.1f}%",
        'Profit': f"{ens_agg_profit:+.1f}u",
        'Status': '✅ PASS' if ens_agg_win_rate >= 0.524 else '❌ FAIL'
    })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
else:
    print("\n⚠️  No models placed bets in 2024 season")

# Go/No-Go decision
print("\n" + "="*80)
print("GO / NO-GO DECISION")
print("="*80)

passing_models = [row for row in summary_data if '✅ PASS' in row['Status']]
failing_models = [row for row in summary_data if '❌ FAIL' in row['Status']]

print(f"\nModels passing (≥52.4% win rate): {len(passing_models)}")
print(f"Models failing (<52.4% win rate): {len(failing_models)}")

if len(passing_models) >= 1 and all(
    float(row['ROI'].replace('%', '').replace('+', '')) > 0
    for row in passing_models
):
    print(f"\n🚀 RECOMMENDATION: PROCEED TO PHASE 3 (Paper Trading)")
    print(f"   At least one model validated with positive ROI")
    print(f"   Next: Paper trade Week 12-16 before going live")
elif len(passing_models) >= 1:
    print(f"\n⚠️  CONDITIONAL: Some models pass but ROI concerns")
    print(f"   Review individual model performance")
    print(f"   Consider paper trading best-performing model only")
else:
    print(f"\n❌ RECOMMENDATION: PAUSE")
    print(f"   No models beat 52.4% threshold in 2024 out-of-sample test")
    print(f"   Edges may have disappeared or were overfit to 2010-2023 data")

print("\n" + "="*80)
print("Backtest complete.")
print("="*80)
