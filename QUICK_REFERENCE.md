# Quick Reference Card - Factor Timing Pipeline

## Commands

```bash
# Production run (ElasticNet)
python run_pipeline.py

# Model comparison (Ridge, ElasticNet, XGBoost)
python run_pipeline.py --compare-models

# Test models
python test_model_selection.py
```

## Key Outputs

### Default Run (`outputs/`)
- `backtest_summary.json` - All performance metrics
- `monthly_returns.csv` - ret_gross, ret_net, turnover
- `factor_allocations.csv` - Factor weights over time
- `stock_weights.csv` - Stock weights over time
- `turnover.csv` - Monthly turnover series
- Plus 7+ plots

### Comparison Run (`outputs/model_comparison/`)
- `model_comparison.csv` - Summary table
- `{model}/` - Full outputs per model
- Comparison plots

## Configuration Quick Edits

**File**: `src/config.py`

```python
# Change model
model_name: str = "elasticnet"  # or "ridge", "xgboost"

# Adjust smoothing (lower = less turnover)
allocation_smoothing: float = 0.7  # 0.0-1.0

# Enable vol targeting
enable_vol_targeting: bool = True
target_vol: float = 0.12  # 12% annualized

# Transaction costs
transaction_cost_bps: float = 10.0  # one-way bps
```

## Key Metrics

| Metric | Good Value | Interpretation |
|--------|------------|----------------|
| Sharpe Ratio | > 1.0 | Risk-adjusted return |
| Sortino Ratio | > 1.5 | Downside risk-adjusted |
| Calmar Ratio | > 1.0 | Return vs max drawdown |
| Win Rate | > 0.55 | Consistency |
| OOS R² | > 0.0 | Predictive power |
| IC | > 0.05 | Factor timing skill |
| Avg Turnover | 20-40% | Trading frequency |

## Troubleshooting

**High Turnover?**
→ Decrease `allocation_smoothing` (e.g., 0.5)

**Low Sharpe?**
→ Check OOS R² and IC
→ Try `--compare-models`

**ElasticNet Warnings?**
→ Normal with small data, safe to ignore

## File Structure

```
qf206_quant/
├── run_pipeline.py          # Main entry point
├── src/
│   ├── config.py            # All configuration
│   ├── backtest.py          # Walk-forward backtest
│   ├── evaluation.py        # Performance metrics
│   ├── plots.py             # All visualizations
│   ├── factor_timing_model.py  # Model training
│   └── ...
├── outputs/                 # Results
├── data/                    # Data cache
└── docs/
    ├── PRODUCTION_README.md
    ├── FINAL_IMPLEMENTATION_SUMMARY.md
    └── ...
```

## Advanced Features

### Allocation Smoothing
```python
# In src/config.py
allocation_smoothing: float = 0.7

# Effect: w_t = 0.7*w_target + 0.3*w_{t-1}
# Lower value = more smoothing, less turnover
```

### Volatility Targeting
```python
# In src/config.py
enable_vol_targeting: bool = True
target_vol: float = 0.12
vol_lookback: int = 12

# Scales returns to maintain 12% annualized vol
# Uses 12-month rolling window
```

## Model Characteristics

| Model | Speed | Interpretability | Overfitting Risk |
|-------|-------|------------------|------------------|
| Ridge | Fast | High | Low |
| ElasticNet | Fast | High | Low |
| XGBoost | Slow | Low | Medium |

## Default Settings (Production)

- **Model**: ElasticNet
- **Regularization**: α = 1.0, l1_ratio = 0.5
- **Transaction Costs**: 10 bps one-way
- **Smoothing**: γ = 0.7
- **Vol Targeting**: Disabled
- **Test Period**: 2018-01-31 onwards
- **Min Training**: 24 months

## Presentation Checklist

✅ Run `python run_pipeline.py --compare-models`
✅ Check `outputs/model_comparison.csv`
✅ Review equity curves in `outputs/model_comparison/`
✅ Examine `outputs/elasticnet/backtest_summary.json`
✅ Show `stock_weights_over_time.png`
✅ Show `equity_curve_gross_vs_net.png`
✅ Discuss transaction cost impact
✅ Highlight OOS R² and IC metrics

## Contact & Support

- Technical docs: `FINAL_IMPLEMENTATION_SUMMARY.md`
- User guide: `PRODUCTION_README.md`
- Original docs: `MODEL_COMPARISON_README.md`
