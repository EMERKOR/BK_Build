# Ball Knower Benchmarks Directory

This directory contains benchmark results comparing Ball Knower predictions
against external sources.

## PredictionTracker Benchmarks

PredictionTracker benchmarks compare BK spreads against the "crowd of models"
consensus from PredictionTracker.

### Input Data

Manually download PredictionTracker NFL prediction CSVs and place them in
`data/external/`:

```
data/external/predictiontracker_nfl_2024.csv
data/external/predictiontracker_nfl_2023.csv
```

### Running Benchmarks

```bash
python src/run_predictiontracker_benchmarks.py \
    --pt_csv data/external/predictiontracker_nfl_2024.csv \
    --output_dir data/benchmarks \
    --outlier_threshold 4.0
```

### Output Files

- `predictiontracker_merged_{season}.csv`: Full merged dataset with BK, PT, and Vegas predictions
- `predictiontracker_summary_{season}.csv`: Summary metrics (MAE, outlier counts, etc.)

### Metrics Computed

- **pt_mae_vs_margin**: Mean absolute error of PredictionTracker consensus vs actual margin
- **bk_mae_vs_margin**: MAE of Ball Knower vs actual margin
- **vegas_mae_vs_margin**: MAE of Vegas closing line vs actual margin
- **bk_vs_pt_diff**: Difference between BK and PT predictions
- **bk_outlier_flag**: Boolean flag for games where |BK - PT| > threshold

### Use Cases

1. **Sanity checking**: Identify games where BK disagrees significantly with consensus
2. **Model validation**: Compare BK's accuracy against the crowd
3. **Edge discovery**: Find potential value bets where BK and PT diverge
4. **Calibration**: Understand BK's biases relative to consensus models
