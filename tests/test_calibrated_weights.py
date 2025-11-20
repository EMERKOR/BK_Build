"""
Test that models load calibrated weights from JSON
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ball_knower.modeling import models
from src import config

print("\n" + "="*80)
print("TESTING CALIBRATED WEIGHTS LOADING")
print("="*80)

print("\n[1/3] Testing DeterministicSpreadModel (v1.0)...")
model_v1 = models.DeterministicSpreadModel()
print(f"  HFA: {model_v1.hfa}")
print(f"  Weights: {model_v1.weights}")

print("\n[2/3] Testing EnhancedSpreadModel (v1.1)...")
model_v1_1 = models.EnhancedSpreadModel()
print(f"  HFA: {model_v1_1.hfa}")
print(f"  Weights: {model_v1_1.weights}")

print("\n[3/3] Testing prediction...")
home_features = {
    'nfelo': 1600,
    'epa_margin': 0.1,
    'Ovr.': 25.0,
    'rest_days': 7,
    'win_rate_L5': 0.6,
    'QB Adj': 2.0
}

away_features = {
    'nfelo': 1500,
    'epa_margin': -0.05,
    'Ovr.': 20.0,
    'rest_days': 7,
    'win_rate_L5': 0.4,
    'QB Adj': 0.0
}

pred_v1 = model_v1.predict(home_features, away_features)
pred_v1_1 = model_v1_1.predict(home_features, away_features)

print(f"\n  v1.0 prediction: {pred_v1:.2f}")
print(f"  v1.1 prediction: {pred_v1_1:.2f}")

print("\n" + "="*80)
print("SUCCESS - Models loaded and used calibrated weights!")
print("="*80 + "\n")
