# Ball Knower - NFL Betting Analytics

A leak-free, modular NFL spread prediction system.

## Project Structure

```
BK_Build/
├── data/
│   ├── current_season/     # PUT YOUR WEEKLY CSV FILES HERE
│   │   ├── nfelo_power_ratings_2025_week_11.csv
│   │   ├── nfelo_strength_of_schedule_2025_week_11.csv
│   │   ├── nfelo_epa_tiers_off_def_2025_week_11.csv
│   │   ├── substack_power_ratings_2025_week_11.csv
│   │   ├── substack_qb_epa_2025_week_11.csv
│   │   └── ... (all your week 11 CSVs)
│   └── reference/          # PUT YOUR REFERENCE FILES HERE
│       ├── nfl_head_coaches.csv
│       └── nfl_AV_data_through_2024.xlsx
├── src/                    # Python modules (we'll build these)
├── notebooks/              # Jupyter notebooks (we'll build these)
└── output/                 # Model predictions and backtest results
```

## Where to Upload Your Files

### Current Season Data → `data/current_season/`
Upload all your nfelo and Substack CSVs here:
- nfelo_power_ratings_2025_week_11.csv
- nfelo_strength_of_schedule_2025_week_11.csv
- nfelo_epa_tiers_off_def_2025_week_11.csv
- nfelo_nfl_win_totals_2025_week_11.csv
- nfelo_nfl_receiving_leaders_2025_week_11.csv
- nfelo_qb_rankings_2025_week_11.csv
- substack_power_ratings_2025_week_11.csv
- substack_qb_epa_2025_week_11.csv
- substack_weekly_proj_ppg_2025_week_11.csv
- substack_weekly_proj_elo_2025_week_11.csv
- substack_qb_elo_archive.csv

### Reference Data → `data/reference/`
Upload these files here:
- nfl_head_coaches.csv
- nfl_AV_data_through_2024.xlsx

## How to Upload

You can upload files by dragging and dropping them into the file explorer, or tell Claude the file path if they're already accessible.

## Model Versions

- **v1.0**: Deterministic baseline (EPA, ratings, structural features)
- **v1.1**: Enhanced with QB/coach/situational features
- **v1.2**: ML correction layer on top of v1.1

## Next Steps

Once you upload the files, Claude will:
1. Inspect the data structure
2. Build team name normalization
3. Create leak-free feature engineering
4. Build and backtest all model versions
5. Generate weekly predictions with edge calculations
