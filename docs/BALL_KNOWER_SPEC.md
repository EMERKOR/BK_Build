# Ball Knower Specification

[Docs Home](README.md) | [Architecture](ARCHITECTURE_OVERVIEW.md) | [Data Sources](DATA_SOURCES.md) | [Feature Tiers](FEATURE_TIERS.md) | [Spec](BALL_KNOWER_SPEC.md) | [Dev Guide](DEVELOPMENT_GUIDE.md)

---

## Purpose

The Ball Knower system is designed to predict NFL game outcomes with increasing sophistication across multiple model versions. The system emphasizes:

- **Transparency**: Clear feature definitions and model logic
- **Reproducibility**: Deterministic results from frozen fixtures
- **Extensibility**: Easy addition of new features and model versions
- **Leakage Prevention**: Strict tier-based feature organization

This specification documents the philosophy, architecture, and workflow of the Ball Knower prediction system.

---

## Model Philosophy

### Evolution Across Versions

The Ball Knower system follows a progressive evolution:

#### v1.0 — Deterministic Baseline
- **Purpose**: Establish a simple, interpretable baseline
- **Approach**: Linear model with core features only
- **Feature Tiers**: Tier 0 (structural) + Tier 1 (core)
- **Output**: Win probability or margin prediction

#### v1.2 — Residual Learning
- **Purpose**: Model the residual error from v1.0
- **Approach**: Train on the difference between actual outcomes and v1.0 predictions
- **Feature Tiers**: Tier 0 + Tier 1 + Tier 2 (market features)
- **Output**: Adjusted win probability or spread prediction

#### v1.3 — Score Prediction
- **Purpose**: Predict actual game scores, not just win probability
- **Approach**: Multi-output regression (home_score, away_score)
- **Feature Tiers**: Tier 0 + Tier 1 + Tier 2
- **Output**: Predicted scores → implied win probability

#### v2.0 — Meta-Model
- **Purpose**: Combine predictions from multiple model versions
- **Approach**: Ensemble or stacking approach using v1.0, v1.2, v1.3 outputs
- **Feature Tiers**: Tier 0 + Tier 1 + Tier 2 + Tier 3 (experimental)
- **Output**: Calibrated ensemble win probability

---

## End-to-End Workflow

### 1. Data Ingestion

Data is loaded from category-first directories using canonical loaders:

- `ball_knower/io/loaders/` contains provider-specific loaders
- Data is validated and normalized into consistent schemas
- Missing data is handled gracefully with fallback logic

See [DATA_SOURCES.md](DATA_SOURCES.md) for details.

### 2. Feature Engineering

Features are computed and organized by tier:

- **Tier 0**: Structural features (always safe)
- **Tier 1**: Core model features (ELO, basic stats)
- **Tier 2**: Market features (betting lines, derived metrics)
- **Tier 3**: Experimental features (advanced analytics)
- **TX**: Forbidden features (leakage risk)

See [FEATURE_TIERS.md](FEATURE_TIERS.md) for details.

### 3. Dataset Construction

Each model version has a dedicated dataset builder:

- `ball_knower/datasets/dataset_builder_v1_0.py`
- `ball_knower/datasets/dataset_builder_v1_2.py`
- `ball_knower/datasets/dataset_builder_v1_3.py`
- `ball_knower/datasets/dataset_builder_v2_0.py`

Dataset builders:
- Load raw data
- Compute required features
- Apply tier-based filtering
- Generate train/val/test splits
- Export to fixtures for testing

### 4. Model Training

Models are trained using version-specific logic:

- `ball_knower/modeling/train_v1_0.py`
- `ball_knower/modeling/train_v1_2.py`
- `ball_knower/modeling/train_v1_3.py`
- `ball_knower/modeling/train_v2_0.py`

Training includes:
- Hyperparameter configuration
- Cross-validation (optional)
- Model serialization
- Performance metrics logging

### 5. Prediction & Backtesting

CLI tools in `src/` enable:

- Single-game predictions
- Multi-week backtests
- Calibration analysis
- Performance reporting

### 6. Evaluation

Results are evaluated using:

- Log loss (calibration)
- Accuracy (classification)
- ROI (betting performance)
- Calibration plots

---

## Supported Model Versions

| Version | Type | Features | Output | Status |
|---------|------|----------|--------|--------|
| v1.0 | Linear Baseline | T0 + T1 | Win Probability | ✅ Implemented |
| v1.2 | Residual Model | T0 + T1 + T2 | Adjusted Win Prob | ✅ Implemented |
| v1.3 | Score Prediction | T0 + T1 + T2 | Scores → Win Prob | ✅ Implemented |
| v2.0 | Meta-Model | T0 + T1 + T2 + T3 | Ensemble Win Prob | 🔨 In Development |

---

## Calibration & Weights Workflow

### Placeholder: Calibration Process

[To be documented]

- How calibration is performed per model version
- Weight initialization strategies
- Validation procedures
- Calibration metrics and thresholds

### Placeholder: Weight Management

[To be documented]

- Where weights are stored
- Versioning strategy
- Loading and updating weights
- Handling model updates

---

## Roadmap

### Short-Term
- Complete v2.0 meta-model implementation
- Add comprehensive calibration workflow
- Expand documentation with examples

### Medium-Term
- Add new feature tiers for advanced analytics
- Implement automated retraining pipeline
- Build web-based prediction interface

### Long-Term
- Multi-sport support (NBA, MLB, etc.)
- Real-time prediction updates
- API for external integrations

---

## References

- [DATA_SOURCES.md](DATA_SOURCES.md) — Data loading and schemas
- [FEATURE_TIERS.md](FEATURE_TIERS.md) — Feature organization
- [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) — System design
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) — Development practices

---

**Status**: This specification is a living document and will evolve as the system grows.
