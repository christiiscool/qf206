# README Updates - What Changed in v2.0

## Overview
The qf206_quant factor timing pipeline has been upgraded to production-ready status with ElasticNet as the default model, comprehensive performance metrics, transaction cost modeling, and advanced portfolio management features.

## Quick Start (Updated)

### Basic Usage
```bash
# Run with default ElasticNet model
python run_pipeline.py

# Run model comparison (Ridge, ElasticNet, XGBoost)
python run_pipeline.py --compare-models
```

### What's New in v2.0

#### 1. ElasticNet is Now Default ✨
- Changed from Ridge to ElasticNet for better feature selection
- Maintains L1+L2 regularization balance
- All three models still available for comparison

#### 2. Stock-Level Allocations 📊
- **New Output**: `stock_weights.csv`
- Shows actual trading positions for each stock
- Validates weight normalization automatically

#### 3. Transaction Costs & Turnover 💰
- Models realistic trading costs (10 bps one-way)
- Tracks monthly turnover
- Reports both gross and net returns
- **New Outputs**: `turnover.csv`, enhanced `monthly_returns.csv`

#### 4. Enhanced Performance Metrics 📈
Now includes 11 comprehensive metrics:
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- CVaR 95%, Win Rate
- Skewness, Kurtosis
- Max Drawdown, Average Turnover
- OOS R², Information Coefficient

#### 5. Professional Visualizations 📉
Six new plots added:
- Stock weights heatmap
- Gross vs net equity curves
- Drawdown series
- Rolling 12-month Sharpe
- Factor return contributions
- Model comparison overlay

#### 6. Advanced Features 🚀
- **Allocation Smoothing**: Reduces turnover (γ=0.7 default)
- **Volatility Targeting**: Optional risk management (disabled by default)

## New Output Structure

### Default Run (`python run_pipeline.py`)
```
outputs/
├── backtest_summary.json       # 11 performance metrics
├── monthly_returns.csv         # ret_gross, ret_net, turnover
├── factor_allocations.csv      # Factor weights
├── stock_weights.csv           # Stock weights [NEW]
├── turnover.csv                # Turnover series [NEW]
└── plots/
    ├── stock_weights_over_time.png [NEW]
    ├── equity_curve_gross_vs_net.png [NEW]
    ├── drawdown_series.png [NEW]
    ├── rolling_sharpe_12m.png [NEW]
    └── ... (7+ total plots)
```

### Comparison Run (`python run_pipeline.py --compare-models`)
```
outputs/model_comparison/
├── model_comparison.csv        # Summary table
├── model_comparison_equity_curves.png [NEW]
├── ridge/                      # Full outputs
├── elasticnet/                 # Full outputs
└── xgboost/                    # Full outputs
```

## Configuration Changes

### Key Parameters (src/config.py)
```python
@dataclass
class FactorTimingConfig:
    model_name: str = "elasticnet"  # Changed from "ridge"
    allocation_smoothing: float = 0.7  # NEW: Reduces turnover
    enable_vol_targeting: bool = False  # NEW: Optional
    target_vol: float = 0.12  # NEW: 12% annualized
    
@dataclass
class BacktestConfig:
    transaction_cost_bps: float = 10.0  # Changed from 5.0
```

## Migration Guide

### From v1.0 to v2.0

**No breaking changes!** Your existing code will continue to work.

**What you get automatically**:
- ElasticNet instead of Ridge (better performance)
- Transaction cost modeling
- Enhanced metrics
- New visualizations
- Stock-level allocations

**Optional upgrades**:
- Enable allocation smoothing (already on by default)
- Enable volatility targeting (edit config)
- Use `--compare-models` flag for presentations

### Backward Compatibility
- Old output files still work
- Column name detection handles both old and new formats
- All existing functions maintain their interfaces

## Performance Expectations

### Typical Metrics (6-stock universe, 2018-2025)
- **Sharpe Ratio**: 0.5 - 1.5
- **Annual Return**: 8% - 15%
- **Max Drawdown**: -15% to -25%
- **Average Turnover**: 20% - 40% monthly
- **OOS R²**: 0.0 - 0.15 (positive = skill)
- **IC**: 0.05 - 0.15 (meaningful timing)

### Transaction Cost Impact
- Typical drag: 0.2% - 0.4% monthly
- Annual impact: ~2.5% - 5%
- Visible in gross vs net equity curves

## Documentation

### New Documentation Files
1. **PRODUCTION_README.md** - Comprehensive production guide
2. **FINAL_IMPLEMENTATION_SUMMARY.md** - Technical details
3. **QUICK_REFERENCE.md** - Quick reference card
4. **OUTPUT_GUIDE.md** - Complete output documentation
5. **DEPLOYMENT_CHECKLIST.md** - Deployment guide
6. **BUGFIX_NOTES.md** - Bug fix history

### Existing Documentation (Updated)
- **MODEL_COMPARISON_README.md** - Still relevant
- **IMPLEMENTATION_SUMMARY.md** - Original implementation
- **QUICK_START.md** - Quick start guide

## Common Tasks

### Change Model
Edit `src/config.py`:
```python
model_name: str = "ridge"  # or "elasticnet", "xgboost"
```

### Reduce Turnover
Edit `src/config.py`:
```python
allocation_smoothing: float = 0.5  # Lower = more smoothing
```

### Enable Volatility Targeting
Edit `src/config.py`:
```python
enable_vol_targeting: bool = True
target_vol: float = 0.12  # 12% annualized
```

### Adjust Transaction Costs
Edit `src/config.py`:
```python
transaction_cost_bps: float = 15.0  # Higher costs
```

## Testing

### Verify Installation
```bash
# Test model selection
python test_model_selection.py

# Should output:
# ✓ ridge trained and predicted successfully
# ✓ elasticnet trained and predicted successfully
# ✓ xgboost trained and predicted successfully
```

### Run Quick Test
```bash
# Run default pipeline
python run_pipeline.py

# Check outputs
ls outputs/
cat outputs/backtest_summary.json
```

## Troubleshooting

### Common Issues

**Q: KeyError 'ret_dynamic_factor'**
A: Fixed in v2.0. Update to latest version.

**Q: High turnover (>60%)**
A: Decrease `allocation_smoothing` to 0.5 or lower

**Q: Low Sharpe ratio (<0.3)**
A: Check OOS R² and IC. Try `--compare-models` to test alternatives.

**Q: ElasticNet convergence warnings**
A: Normal with small datasets. Safe to ignore.

## What's Next

### Potential Enhancements
- Hyperparameter tuning via grid search
- Ensemble methods (model averaging)
- Factor-specific models
- Additional regime features
- Risk parity weighting

### Research Ideas
- Test on larger universes
- Longer backtesting periods
- Alternative rebalancing frequencies
- Sector-neutral constraints

## Support

- **Technical Issues**: See BUGFIX_NOTES.md
- **Usage Questions**: See QUICK_REFERENCE.md
- **Output Questions**: See OUTPUT_GUIDE.md
- **Deployment**: See DEPLOYMENT_CHECKLIST.md

## Version History

### v2.0.0 (2026-03-05) - Production Release
- ElasticNet as default model
- Transaction cost modeling
- Enhanced metrics (11 total)
- Stock-level allocations
- 6 new visualizations
- Allocation smoothing
- Volatility targeting
- Comprehensive documentation

### v1.0.0 (2026-03-04) - Initial Release
- Ridge regression baseline
- Basic factor timing
- 4 performance metrics
- Factor allocations
- Basic visualizations

## Credits

Developed for QF206 Quantitative Finance course.

## License

[Your License Here]
