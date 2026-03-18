# Production Factor Timing Pipeline - Complete Guide

## Overview

The qf206_quant project now features a production-ready factor timing strategy with **ElasticNet as the default model**, comprehensive performance metrics, transaction cost modeling, and advanced portfolio management features.

## Quick Start

### Run Default Model (ElasticNet)
```bash
python run_pipeline.py
```

### Run Model Comparison
```bash
python run_pipeline.py --compare-models
```

## What's New

### 1. ElasticNet is Now Default
- Changed from Ridge to ElasticNet for better feature selection
- Maintains L1+L2 regularization balance (l1_ratio=0.5)
- All three models (Ridge, ElasticNet, XGBoost) remain available for comparison

### 2. Stock-Level Allocations (PART 2)
- **New Output**: `stock_weights.csv` - Final trading weights for each stock
- Tracks month-by-month portfolio composition
- Validates weight normalization (sum ≈ 1.0)
- Logs warnings for NaN or abnormal weights

### 3. Transaction Costs & Turnover (PART 3)
- **Turnover Calculation**: `turnover_t = Σ|w_{i,t} - w_{i,t-1}|`
- **Transaction Costs**: 10 bps one-way (configurable in `BacktestConfig`)
- **Net Returns**: `ret_net = ret_gross - (tc_bps/10000) * turnover`
- **New Outputs**:
  - `turnover.csv` - Monthly turnover series
  - `monthly_returns.csv` now includes: ret_gross, ret_net, turnover

### 4. Enhanced Metrics (PART 3)
All backtest summaries now include:

**Performance Metrics**:
- Annual Return, Volatility, Sharpe Ratio
- **Sortino Ratio** (downside deviation)
- **Calmar Ratio** (CAGR / |MaxDD|)
- **CVaR 95%** (Value at Risk)
- **Win Rate** (% positive months)
- Skewness, Kurtosis
- Max Drawdown

**Prediction Quality**:
- Out-of-Sample R²
- Information Coefficient (IC)
- Average Turnover

### 5. New Visualizations (PART 4)

**Default Run Plots** (saved to `outputs/`):
1. `stock_weights_over_time.png` - Heatmap of stock allocations
2. `equity_curve_gross_vs_net.png` - Impact of transaction costs
3. `drawdown_series.png` - Underwater plot
4. `rolling_sharpe_12m.png` - Time-varying risk-adjusted performance
5. `factor_weights_over_time.png` - Factor allocation heatmap
6. `cumulative_dynamic_factor_portfolio.png` - Main equity curve

**Comparison Mode Plots** (saved to `outputs/model_comparison/`):
- `model_comparison_equity_curves.png` - Overlay of all models
- `model_comparison_metrics.png` - Bar charts of key metrics
- Individual model directories with full plot suites

### 6. Advanced Features (PART 5)

#### Option A: Allocation Smoothing (Implemented)
Reduces turnover by smoothing weight transitions:
```
w_t = γ * w_target + (1-γ) * w_{t-1}
```
- **Config**: `FactorTimingConfig.allocation_smoothing = 0.7` (default)
- Set to 1.0 to disable smoothing
- Lower values = more smoothing, less turnover

#### Option B: Volatility Targeting (Implemented)
Scales portfolio exposure to maintain target volatility:
- **Config**: 
  - `enable_vol_targeting = False` (default, disabled)
  - `target_vol = 0.12` (12% annualized)
  - `vol_lookback = 12` (months)
- Scaling capped at [0.5, 2.0] to avoid extreme leverage
- Uses rolling realized volatility

**To Enable Vol Targeting**:
Edit `src/config.py`:
```python
@dataclass
class FactorTimingConfig:
    enable_vol_targeting: bool = True
    target_vol: float = 0.12
```

## Output Structure

### Default Run
```
outputs/
├── backtest_summary.json       # All performance metrics
├── monthly_returns.csv         # ret_gross, ret_net, turnover
├── factor_allocations.csv      # Factor weights over time
├── stock_weights.csv           # Stock weights over time
├── turnover.csv                # Monthly turnover series
└── plots/
    ├── stock_weights_over_time.png
    ├── equity_curve_gross_vs_net.png
    ├── drawdown_series.png
    ├── rolling_sharpe_12m.png
    └── ...
```

### Comparison Mode
```
outputs/
├── model_comparison.csv        # Summary table
└── model_comparison/
    ├── model_comparison_equity_curves.png
    ├── model_comparison_metrics.png
    ├── ridge/
    │   ├── backtest_summary.json
    │   ├── monthly_returns.csv
    │   ├── factor_allocations.csv
    │   ├── stock_weights.csv
    │   ├── turnover.csv
    │   └── plots...
    ├── elasticnet/
    │   └── (same structure)
    └── xgboost/
        └── (same structure)
```

## Configuration Reference

### Key Parameters

**Model Selection** (`src/config.py`):
```python
@dataclass
class FactorTimingConfig:
    model_name: str = "elasticnet"  # "ridge", "elasticnet", "xgboost"
    regularization: float = 1.0
    allocation_smoothing: float = 0.7  # 0.0-1.0, higher = less smoothing
    enable_vol_targeting: bool = False
    target_vol: float = 0.12
    vol_lookback: int = 12
```

**Transaction Costs** (`src/config.py`):
```python
@dataclass
class BacktestConfig:
    transaction_cost_bps: float = 10.0  # one-way cost in basis points
```

## Comparison Table Metrics

The `model_comparison.csv` includes:

| Metric | Description |
|--------|-------------|
| annual_return | Annualized return (net of costs) |
| annual_volatility | Annualized volatility |
| sharpe_ratio | Return / Volatility |
| sortino_ratio | Return / Downside Deviation |
| calmar_ratio | CAGR / |Max Drawdown| |
| max_drawdown | Peak-to-trough decline |
| cvar_95 | 95% Conditional Value at Risk |
| win_rate | Fraction of positive months |
| avg_turnover | Average monthly turnover |
| oos_r2 | Out-of-sample R² of predictions |
| ic | Information Coefficient |

## Walk-Forward Backtest Details

- **Training Window**: Expanding (all data before month t)
- **Minimum Training**: 24 months
- **Test Period**: 2018-01-31 onwards
- **Rebalancing**: Monthly (month-end)
- **No Look-Ahead Bias**: Predictions use only past data
- **Reproducible**: `random_state=42` throughout

## Performance Validation

### Weight Checks
The backtest automatically validates:
- Weight sum ≈ 1.0 (logs warning if |sum - 1.0| > 0.01)
- No NaN weights (logs warning if detected)
- Turnover reasonableness

### Prediction Quality
- **OOS R² > 0**: Model has predictive power
- **|IC| > 0.05**: Meaningful factor timing skill
- **|IC| > 0.10**: Strong factor timing skill

## Troubleshooting

### High Turnover
If turnover is excessive:
1. Increase `allocation_smoothing` (e.g., 0.5 instead of 0.7)
2. Increase `regularization` to reduce model sensitivity
3. Consider adding turnover penalties to allocation logic

### Low Sharpe Ratio
If risk-adjusted returns are poor:
1. Check prediction quality (OOS R², IC)
2. Try different models via `--compare-models`
3. Review factor allocations for concentration
4. Enable volatility targeting

### Convergence Warnings (ElasticNet)
- Normal with small datasets
- Increase `max_iter` in `src/factor_timing_model.py` if persistent
- Usually safe to ignore

## Next Steps for Research

1. **Hyperparameter Tuning**: Grid search for optimal regularization
2. **Factor Engineering**: Add new regime features
3. **Ensemble Methods**: Combine multiple models
4. **Risk Management**: Add position limits, sector constraints
5. **Alternative Objectives**: Maximize Sharpe instead of returns

## Code Quality

✅ No breaking changes to existing pipeline
✅ Backward compatible with previous outputs
✅ All functions maintain consistent interfaces
✅ Comprehensive logging throughout
✅ Reproducible results (random_state=42)
✅ No look-ahead bias
✅ Modular design for easy extension

## Support Files

- `MODEL_COMPARISON_README.md` - Original model comparison documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `QUICK_START.md` - Quick reference guide
- `test_model_selection.py` - Unit tests for model selection

## Citation

When presenting results, cite:
- Model: ElasticNet (L1+L2 regularization, α=1.0, l1_ratio=0.5)
- Transaction Costs: 10 bps one-way
- Allocation Smoothing: γ=0.7
- Test Period: 2018-01-31 to present
- Universe: 6 stocks (AAPL, MSFT, NVDA, JPM, XOM, UNH)
