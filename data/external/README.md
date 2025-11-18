# External Data Directory

Place manually downloaded external datasets here.

## PredictionTracker

Download NFL prediction CSV files from PredictionTracker and save them here:

- `predictiontracker_nfl_2024.csv`
- `predictiontracker_nfl_2023.csv`

Expected CSV format:
- Home/Away team columns
- Vegas line (optional)
- Prediction average or individual model predictions
- Prediction standard deviation (optional)

The loader will auto-detect column names and normalize team names using
Ball Knower's canonical team mapping.

## Other External Sources

Place other external benchmarking data here as needed (e.g., 538, Vegas archives, etc.)
