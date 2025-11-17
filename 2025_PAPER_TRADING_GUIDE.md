# Ball Knower 2025 Paper Trading Guide

**Validated Edge**: Favorites 5+ Point Strategy (54.6% win rate, 306 bets, 2020-2024)

This guide shows you how to paper trade the strategy for the 2025 NFL season.

---

## Quick Start

### 1. Update Your Data (Before Each Week)

Make sure you have the latest 2025 data:

```bash
# Update schedules.parquet with Week N games
# Update team_week_epa_2013_2024.csv with latest EPA data
# (Instructions for data updates depend on your data source)
```

### 2. Generate Weekly Picks

Run this command each week **BEFORE** games start:

```bash
python generate_weekly_picks_2025.py --week 1
```

This will:
- Load all historical data
- Train model on 2013-2024
- Generate predictions for Week 1
- Identify "Favorites 5+ Edge" bets
- Save picks to `picks_2025/` folder

### 3. Review Picks

Check the generated files:

```bash
picks_2025/
├── week_1_summary.txt          # Human-readable picks
├── week_1_bets.csv            # Bets only (CSV format)
└── week_1_all_predictions.csv # All games with predictions
```

### 4. Record Picks in Tracker

Open `PAPER_TRADING_TRACKER.md` and fill in:
- Game details
- Bet team and line
- Model edge

**DO THIS BEFORE GAMES START** - No cheating!

### 5. Wait for Games to Complete

Let the week play out. Do not bet real money.

### 6. Update Results

After all games finish:
- Update "Result" column (final score)
- Mark "Win?" (did bet cover?)
- Calculate units (+1.0 for win, -1.1 for loss)
- Update weekly totals

### 7. Repeat for Weeks 2-4

Run steps 2-6 for each week.

### 8. Evaluate After Week 4

Review Phase 1 results:

| Outcome | Win Rate | Action |
|---------|----------|--------|
| **Excellent** | ≥55% | Proceed to small stakes (Week 5+) |
| **Good** | 52-54% | Continue paper trading |
| **Poor** | <52% | Pause and reassess strategy |

---

## Example Workflow (Week 1)

### Tuesday Before Week 1

**Step 1**: Generate picks

```bash
$ python generate_weekly_picks_2025.py --week 1

================================================================================
Ball Knower - 2025 Weekly Picks Generator
================================================================================

📅 Generating picks for 2025 Week 1
⏰ Generated: 2025-09-03 14:30:00

...

📋 BETS FOR WEEK 1
================================================================================

🎯 BET #1
   Game: BUF @ NYJ
   Date: 2025-09-09 20:20

   Vegas Line: BUF -3.0
   Model Prediction: BUF should win by +9.2
   Model Edge: 6.2 points

   → BET: BUF -3.0
   Confidence: HIGH (validated 54.6% win rate)

================================================================================
✅ Pick generation complete!
================================================================================

📋 1 bet(s) identified for Week 1
```

**Step 2**: Record in tracker

Update `PAPER_TRADING_TRACKER.md`:

```markdown
## Week 1

| # | Game | Bet | Line | Model Edge | Result | Win? | Units |
|---|------|-----|------|------------|--------|------|-------|
| 1 | BUF @ NYJ | BUF | -3.0 | 6.2 pts | TBD | - | - |

**Week 1 Totals**: 0-0, 0.0 units, 0.0% win rate
```

### Monday After Week 1

Game result: BUF 24, NYJ 20 (BUF wins by 4)

Update tracker:

```markdown
## Week 1

| # | Game | Bet | Line | Model Edge | Result | Win? | Units |
|---|------|-----|------|------------|--------|------|-------|
| 1 | BUF @ NYJ | BUF | -3.0 | 6.2 pts | BUF by 4 | ✅ | +1.0 |

**Week 1 Totals**: 1-0, +1.0 units, 100% win rate
```

BUF won by 4, covering the -3.0 spread → WIN!

---

## Command Options

### Basic Usage

```bash
python generate_weekly_picks_2025.py --week 1
```

### Save Trained Model (Faster for Multiple Weeks)

```bash
# First time: train and save
python generate_weekly_picks_2025.py --week 1 --save-model

# Later weeks: load saved model
python generate_weekly_picks_2025.py --week 2 --load-model models/ball_knower_v2_trained_20250903.pkl
```

This speeds up pick generation since you don't retrain each week.

---

## Understanding the Output

### Summary File (`week_N_summary.txt`)

```
================================================================================
Ball Knower - 2025 Week 1 Picks
================================================================================

Generated: 2025-09-03 14:30:00
Strategy: Favorites 5+ Edge (54.6% validated win rate)

Total games: 14
Strategy bets: 1

================================================================================
PICKS FOR WEEK 1
================================================================================

BET #1
  Game: BUF @ NYJ
  Date: 2025-09-09 20:20

  BET: BUF -3.0
  Model Edge: 6.2 points
  Confidence: HIGH
```

### CSV File (`week_N_bets.csv`)

| game_id | away_team | home_team | vegas_spread | model_edge | bet_team | bet_line |
|---------|-----------|-----------|--------------|------------|----------|----------|
| 2025_01_BUF_NYJ | BUF | NYJ | -3.0 | +6.2 | BUF | -3.0 |

---

## Strategy Recap

### What We Bet
- **Favorites only** with ≥5 point edge
- Model says "favorite should win by MORE than Vegas"
- 1-2 bets per week on average

### Why It Works
- Vegas overadjusts favorites (public betting patterns)
- Creates underpriced favorites
- Model identifies these cases

### Validated Results (2020-2024)
- 306 bets
- 54.6% win rate
- +4.6% ROI

### Expected 2025 Volume
- ~60-65 bets total season
- ~4 bets per week during busy weeks
- Some weeks: 0 bets (no qualifying games)

---

## Common Questions

### Q: What if no bets are generated for a week?

**A**: That's normal! Some weeks have no games meeting criteria. The strategy is selective.

### Q: Can I bet on multiple games in one week?

**A**: Yes, if multiple games qualify. Each is independent.

### Q: What if I disagree with a pick?

**A**: Paper trading means you don't have to actually bet. Just track what the model says and see how it performs.

### Q: When can I start betting real money?

**A**: **NOT before Week 5**. You MUST paper trade Weeks 1-4 first to validate.

### Q: What bet size should I use (when ready)?

**A**: Start with 0.5% of bankroll. Example:
- $10,000 bankroll → $50/bet
- $5,000 bankroll → $25/bet
- $1,000 bankroll → $5/bet

### Q: What's the stop-loss?

**A**: If you're ever down 10 units cumulative, STOP betting and reassess.

---

## Troubleshooting

### Error: "No games found for 2025 Week N"

**Solution**: Update `schedules.parquet` with 2025 data

### Error: "ModuleNotFoundError: No module named 'feature_engineering_v2'"

**Solution**: Make sure you're in the BK_Build directory:
```bash
cd /path/to/BK_Build
python generate_weekly_picks_2025.py --week 1
```

### Error: "KeyError: 'spread_line'"

**Solution**: 2025 games don't have Vegas lines yet. Wait for lines to be posted (usually Tuesday-Wednesday before games).

---

## Files You'll Work With

### Created by You
- `PAPER_TRADING_TRACKER.md` - Where you record all picks and results

### Created Each Week
- `picks_2025/week_N_summary.txt` - Readable summary
- `picks_2025/week_N_bets.csv` - Bet data
- `picks_2025/week_N_all_predictions.csv` - All game predictions

### One-Time Setup
- `generate_weekly_picks_2025.py` - Script to generate picks
- Data files: `schedules.parquet`, `team_week_epa_2013_2024.csv`, etc.

---

## Phase 1 Success Criteria

After 4 weeks of paper trading, you should have:

✅ **12-16 total bets** (3-4 per week average)
✅ **≥52% win rate** (ideally ≥55%)
✅ **Positive ROI** (>0%)
✅ **Consistent with validation** (54.6% historical)

If you hit these targets → Proceed to Phase 2 (small stakes)

If not → Pause and figure out why (market changed? Data issues? Bad luck?)

---

## Next Steps After Validation

If Week 4 results are good (≥55% win rate):

1. **Start Small** (Weeks 5-8)
   - Bet 0.5% of bankroll per bet
   - Continue tracking meticulously
   - Check in every 4 weeks

2. **Scale Gradually** (Weeks 9-12)
   - Increase to 0.75% if still profitable
   - Maintain stop-loss discipline

3. **Full Stakes** (Weeks 13+, Playoffs)
   - Up to 1.0% per bet if consistently profitable
   - Never exceed 2% on single bet

---

## Remember

This is a **REAL edge** (validated on 306 bets), but it's **SMALL** (2.2% above vig).

### Do's ✅
- Paper trade Weeks 1-4 first
- Track every single bet
- Respect the stop-loss
- Start small when ready
- Be patient with variance

### Don'ts ❌
- Skip validation phase
- Bet large amounts immediately
- Chase losses
- Bet on non-favorites
- Ignore bad results

---

Good luck, and may the odds be ever in your favor! 🎯

**Questions?** Review `BALL_KNOWER_V2_RESULTS.md` for full analysis.
