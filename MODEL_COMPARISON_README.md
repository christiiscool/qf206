# Model Comparison Framework

This document describes the multi-model factor timing framework added to the quantitative finance pipeline.

## Overview

The pipeline now supports comparing multiple machine learning models for factor timing:
- **Ridge Regression**: L2 regularized linear regression (baseline)
- **ElasticNet**: L1 + L2 regularized linear regression
- **XGBoost**: Gradient boosted decision trees

All models use `MultiOutputRegressor` to predict returns for all factors simultaneously.

## Configuration

The model is selected via the `FactorTimingConfig.model_name` parameter in `src/config.py`:

```python
@dataclass
class FactorTimingConfig:
    model_name: str = "ridge"  # Options: "ridge", "elasticnet", "xgboost"
    regularization: float = 1.0
    # ... other parameters
```

## Running Single Model

To run the pipeline with a specific model, edit `src/config.py` and set `model_name`, then run:

```bash
python run_pipeline.py
```

## Running Model Comparison

To compare all three models in a single run:

```bash
python run_model_comparison.py
```

This will:
1. Run the full walk-forward backtest for each model
2. Save model-specific outputs to separate directories:
   - `outputs/ridge/`
   - `outputs/elasticnet/`
   - `outputs/xgboost/`
3. Generate a comparison table: `outputs/model_comparison.csv`

## Output Structure

```
outputs/
├── model_comparison.csv          # Summary comparison table
├── ridge/
│   ├── backtest_summary.json
│   ├── monthly_returns.csv
│   ├── factor_allocations.csv
│   └── plots...
├── elasticnet/
│   └── (same structure)
└── xgboost/
    └── (same structure)
```

## Comparison Metrics

The comparison table includes:

### Portfolio Performance
- **Annual Return**: Annualized portfolio return
- **Annual Volatility**: Annualized return volatility
- **Sharpe Ratio**: Risk-adjusted return (return/volatility)
- **Max Drawdown**: Maximum peak-to-trough decline

### Prediction Quality
- **OOS R²**: Out-of-sample R² of factor return predictions
- **IC**: Information Coefficient (correlation between predicted and realized factor returns)

## Model Details

### Ridge Regression
```python
Ridge(alpha=cfg.regularization, random_state=42)
```
- Linear model with L2 penalty
- Fast training, interpretable
- Good baseline for factor timing

### ElasticNet
```python
ElasticNet(alpha=cfg.regularization, l1_ratio=0.5, max_iter=5000, random_state=42)
```
- Combines L1 and L2 penalties
- Can perform feature selection
- l1_ratio=0.5 balances both penalties

### XGBoost
```python
XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
```
- Non-linear gradient boosted trees
- Can capture complex regime interactions
- More prone to overfitting with limited data

## Walk-Forward Backtest

All models use the same expanding window backtest:
- Minimum 24 months training data
- Train on all data before month t
- Predict factor returns for month t
- No look-ahead bias
- Reproducible with `random_state=42`

## Code Changes

### Modified Files
- `src/config.py`: Added `model_name` parameter
- `src/factor_timing_model.py`: Dynamic model selection
- `src/backtest.py`: Added prediction metrics tracking
- `src/evaluation.py`: Returns stats dict for comparison

### New Files
- `src/model_comparison.py`: Model comparison orchestration
- `run_model_comparison.py`: Comparison pipeline script
- `MODEL_COMPARISON_README.md`: This documentation

## Example Usage

```python
from src.config import get_default_config
from src.model_comparison import run_model_comparison

cfg = get_default_config()
# ... load data ...

# Compare specific models
comparison_df = run_model_comparison(
    cfg, 
    labeled_panel, 
    spy_daily,
    models=["ridge", "xgboost"]
)

print(comparison_df)
```

## Notes

- All models maintain the same interface for compatibility
- The factor allocation logic remains unchanged
- Transaction costs and constraints are applied consistently
- Each model is trained independently in the walk-forward loop
