# Quick Start Guide: Model Comparison

## Installation

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Test the Implementation

Verify all models work correctly:
```bash
python test_model_selection.py
```

Expected output:
```
Testing ridge...
✓ ridge trained and predicted successfully
Testing elasticnet...
✓ elasticnet trained and predicted successfully
Testing xgboost...
✓ xgboost trained and predicted successfully
All models tested successfully!
```

## Run Model Comparison

Execute the full comparison pipeline:
```bash
python run_model_comparison.py
```

This will:
1. Download/load data for 6 stocks + SPY
2. Build monthly features
3. Run walk-forward backtest for Ridge, ElasticNet, and XGBoost
4. Generate comparison table and plots

**Runtime**: ~2-5 minutes depending on data availability

## View Results

After completion, check:

### Comparison Table
```bash
cat outputs/model_comparison.csv
```

### Comparison Plots
- `outputs/model_comparison_metrics.png` - Bar charts of all metrics
- `outputs/cumulative_returns_comparison.png` - Equity curves overlay

### Model-Specific Results
```
outputs/
├── ridge/
│   ├── backtest_summary.json
│   ├── monthly_returns.csv
│   ├── factor_allocations.csv
│   └── *.png (plots)
├── elasticnet/
│   └── (same structure)
└── xgboost/
    └── (same structure)
```

## Run Single Model

To run just one model (faster for testing):

1. Edit `src/config.py`:
```python
@dataclass
class FactorTimingConfig:
    model_name: str = "xgboost"  # Change this
    # ...
```

2. Run:
```bash
python run_pipeline.py
```

## Interpret Results

### Portfolio Performance Metrics

- **Annual Return**: Higher is better (but watch volatility)
- **Sharpe Ratio**: Risk-adjusted return, >1.0 is good, >2.0 is excellent
- **Max Drawdown**: Smaller (less negative) is better

### Prediction Quality Metrics

- **OOS R²**: Measures prediction accuracy
  - Positive values indicate predictive power
  - Negative values indicate worse than naive mean prediction
  
- **IC (Information Coefficient)**: Correlation between predictions and actuals
  - Range: -1 to +1
  - |IC| > 0.05 is meaningful in factor timing
  - |IC| > 0.10 is strong

## Troubleshooting

### Data Download Issues
If data download fails, the pipeline will use cached data from `data/raw/`

### Memory Issues
The 6-stock universe is intentionally small. If you encounter memory issues:
- Close other applications
- Reduce the date range in `src/config.py`

### Model Training Warnings
- ElasticNet may show convergence warnings with small datasets (safe to ignore)
- XGBoost may show warnings about feature names (cosmetic, safe to ignore)

## Customization

### Change Stock Universe
Edit `src/config.py`:
```python
demo_tickers: List[str] = field(
    default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]
)
```

### Adjust Model Hyperparameters
Edit `src/factor_timing_model.py` in the `train_factor_timing_model()` function

### Change Date Range
Edit `src/config.py`:
```python
@dataclass
class DataConfig:
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"

@dataclass
class BacktestConfig:
    test_start: str = "2020-01-31"
```

## Next Steps

1. ✅ Run `test_model_selection.py` to verify setup
2. ✅ Run `run_model_comparison.py` for full comparison
3. 📊 Analyze results in `outputs/model_comparison.csv`
4. 📈 Review equity curves and factor allocations
5. 🔧 Experiment with different hyperparameters
6. 📝 Document findings for your research

## Support

For detailed documentation, see:
- `MODEL_COMPARISON_README.md` - Full feature documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `README.md` - Original project documentation
