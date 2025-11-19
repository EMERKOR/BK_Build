# BK_Build Documentation

[Docs Home](README.md) | [Architecture](ARCHITECTURE_OVERVIEW.md) | [Data Sources](DATA_SOURCES.md) | [Feature Tiers](FEATURE_TIERS.md) | [Spec](BALL_KNOWER_SPEC.md) | [Dev Guide](DEVELOPMENT_GUIDE.md)

---

## Overview

Welcome to the BK_Build documentation. This project implements a multi-version NFL game prediction system with a focus on:

- **Deterministic baseline models** (v1.0)
- **Residual learning** (v1.2)
- **Score prediction** (v1.3)
- **Meta-models** (v2.0)

The system is designed for extensibility, reproducibility, and clean separation of concerns.

---

## What is BK_Build?

BK_Build is a structured framework for building, training, and evaluating NFL prediction models. It provides:

- Canonical data loading pipelines
- Feature engineering organized by tier
- Multiple model architectures
- Backtesting and evaluation infrastructure
- CLI tools for running predictions and analysis

---

## Project Organization

The codebase is organized as follows:

### Core Package: `ball_knower/`

This is the **canonical source** of all production code:

- **`ball_knower/config/`** — Configuration management and settings
- **`ball_knower/features/`** — Feature definitions organized by tier
- **`ball_knower/modeling/`** — Model implementations and training logic
- **`ball_knower/io/loaders/`** — Data loading and preprocessing
- **`ball_knower/datasets/`** — Dataset builders for each model version

### CLI Scripts: `src/`

Command-line tools for running backtest operations, predictions, and analysis.

### Data: `data/`

Raw and processed data files organized by category and provider.

### Tests: `tests/`

Fixture-based unit tests for validation and regression detection.

### Documentation: `docs/`

You are here! Comprehensive documentation for developers and users.

---

## Documentation Index

- **[BALL_KNOWER_SPEC.md](BALL_KNOWER_SPEC.md)** — Detailed specification of the Ball Knower system
- **[DATA_SOURCES.md](DATA_SOURCES.md)** — Data sources, naming conventions, and schemas
- **[FEATURE_TIERS.md](FEATURE_TIERS.md)** — Feature tier system and leakage prevention
- **[ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)** — System architecture and design principles
- **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** — Guidelines for contributing and extending the system

---

## Quick Start

1. **Installation**: Set up your environment (see [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md))
2. **Data Setup**: Review [DATA_SOURCES.md](DATA_SOURCES.md) for data requirements
3. **Run Tests**: Execute fixture-based tests to validate setup
4. **Backtest**: Use CLI tools in `src/` to run backtests

---

## Key Principles

- **Category-First Data Organization**: Data is organized by category (e.g., `elo`, `odds`, `stats`) with provider subdirectories
- **Feature Tier System**: Features are categorized by tier (0-3, TX) to prevent leakage
- **Fixture-Based Testing**: All tests use frozen fixtures to ensure reproducibility
- **Canonical Code Location**: All production code lives in `ball_knower/`

---

## Contributing

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for contribution guidelines and development practices.

---

## License

[Add license information here]

---

**Next Steps**: Explore the other documentation files to dive deeper into specific aspects of the system.
