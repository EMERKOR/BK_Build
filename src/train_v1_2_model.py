"""
Train Ball Knower v1.2 Model

CLI script to train the v1.2 residual model.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))

    from ball_knower.models.v1_2.train import train_v1_2

    # Train model with default settings
    model_path = train_v1_2()

    print(f"\n✓ Model training complete!")
    print(f"✓ Model saved to: {model_path}")
