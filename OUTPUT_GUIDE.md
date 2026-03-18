# Complete Output Guide

## Default Run: `python run_pipeline.py`

### CSV Files (outputs/)
```
backtest_summary.json       # All performance metrics in JSON format
monthly_returns.csv         # Columns: ret_gross, ret_net, turnover
factor_allocations.csv      # Factor weights over time (4 factors × months)
stock_weights.csv           # Stock weights over time (6 stocks × months) [NEW]
turnover.csv                # Monthly turnover series [NEW]
```

### Plots (outputs/)
```
# Original plots
cumulative_dynamic_factor_portfolio.png
factor_weights_over_time.png
demo_universe_cumulative_forward_returns.png

# New plots (Part 4)
stock_weights_over_time.png              # Heatmap of stock allocations
equity_curve_gross_vs_net.png            # Gross vs net returns
drawdown_series.png                      # Underwater plot
rolling_sharpe_12m.png                   # Rolling 12-month Sharpe
```

### backtest_summary.json Structure
```json
{
  "dynamic_factor_portfolio": {
    "ann_return": 0.123,           # Annualized return (net)
    "ann_vol": 0.156,              # Annualized volatility
    "sharpe": 0.789,               # Sharpe ratio
    "sortino": 1.234,              # Sortino ratio [NEW]
    "calmar": 0.567,               # Calmar ratio [NEW]
    "cvar_95": -0.045,             # 95% CVaR [NEW]
    "win_rate": 0.583,             # Win rate [NEW]
    "skewness": -0.123,            # Skewness [NEW]
    "kurtosis": 2.345,             # Kurtosis [NEW]
    "max_drawdown": -0.234,        # Max drawdown
    "gross_ann_return": 0.145,     # Gross annual return [NEW]
    "gross_ann_vol": 0.156,        # Gross volatility [NEW]
    "gross_sharpe": 0.929,         # Gross Sharpe [NEW]
    ... (all gross metrics) ...
    "oos_r2": 0.067,               # Out-of-sample R²
    "ic": 0.089,                   # Information Coefficient
    "avg_turnover": 0.287          # Average monthly turnover [NEW]
  }
}
```

### monthly_returns.csv Structure
```csv
month_end,ret_gross,ret_net,turnover
2018-01-31,0.0234,0.0205,0.3456
2018-02-28,-0.0123,-0.0145,0.2789
...
```

### stock_weights.csv Structure
```csv
month_end,AAPL,MSFT,NVDA,JPM,XOM,UNH
2018-01-31,0.1234,0.2345,0.1567,0.1890,0.1456,0.1508
2018-02-28,0.1456,0.2123,0.1789,0.1678,0.1567,0.1387
...
```

## Comparison Run: `python run_pipeline.py --compare-models`

### Root Level (outputs/)
```
demo_universe_cumulative_forward_returns.png  # Universe plot (once)
```

### Comparison Directory (outputs/model_comparison/)
```
model_comparison.csv                    # Summary table
model_comparison_equity_curves.png      # Overlay of all models [NEW]
model_comparison_metrics.png            # Bar charts
cumulative_returns_comparison.png       # Alternative overlay
```

### Per-Model Directories (outputs/model_comparison/{model}/)
Each of ridge/, elasticnet/, xgboost/ contains:

```
# CSV files
backtest_summary.json
monthly_returns.csv
factor_allocations.csv
stock_weights.csv          [NEW]
turnover.csv               [NEW]

# Plots
cumulative_dynamic_factor_portfolio.png
factor_weights_over_time.png
stock_weights_over_time.png              [NEW]
equity_curve_gross_vs_net.png            [NEW]
drawdown_series.png                      [NEW]
rolling_sharpe_12m.png                   [NEW]
```

### model_comparison.csv Structure
```csv
model,annual_return,annual_volatility,sharpe_ratio,sortino_ratio,calmar_ratio,max_drawdown,cvar_95,win_rate,avg_turnover,oos_r2,ic
ridge,0.123,0.156,0.789,1.234,0.567,-0.234,-0.045,0.583,0.287,0.067,0.089
elasticnet,0.145,0.167,0.868,1.345,0.678,-0.212,-0.038,0.600,0.245,0.078,0.095
xgboost,0.134,0.178,0.753,1.123,0.589,-0.227,-0.042,0.567,0.312,0.045,0.067
```

## Data Files (data/)

### Raw Data (data/raw/)
```
daily_ohlcv.parquet         # Daily OHLCV for 6 stocks
daily_spy.parquet           # Daily SPY data
sp500_constituents.csv      # Universe definition
data_sources.json           # Data provenance
```

### Processed Data (data/processed/)
```
monthly_panel.parquet       # Monthly feature panel with labels
```

## Plot Descriptions

### 1. cumulative_dynamic_factor_portfolio.png
- Line chart of cumulative portfolio growth
- Shows overall strategy performance
- Net returns (after transaction costs)

### 2. factor_weights_over_time.png
- Heatmap: factors (rows) × time (columns)
- Color scale: -0.6 to +0.6 (max_factor_weight)
- Shows factor allocation dynamics

### 3. stock_weights_over_time.png [NEW]
- Heatmap: stocks (rows) × time (columns)
- Color scale: centered at 0
- Shows actual trading positions

### 4. equity_curve_gross_vs_net.png [NEW]
- Two lines: gross returns vs net returns
- Shows transaction cost impact
- Gap between lines = cumulative TC drag

### 5. drawdown_series.png [NEW]
- Area chart of drawdowns over time
- Shows risk periods
- Helps identify max drawdown timing

### 6. rolling_sharpe_12m.png [NEW]
- Line chart of rolling 12-month Sharpe ratio
- Shows time-varying risk-adjusted performance
- Horizontal line at Sharpe=1 for reference

### 7. model_comparison_equity_curves.png [NEW]
- Three lines: ridge, elasticnet, xgboost
- Overlay comparison of cumulative returns
- Shows relative model performance

### 8. demo_universe_cumulative_forward_returns.png
- Six lines: one per stock
- Shows individual stock performance
- Educational/diagnostic plot

## File Sizes (Approximate)

```
backtest_summary.json       ~2 KB
monthly_returns.csv         ~5 KB (80 months)
factor_allocations.csv      ~4 KB
stock_weights.csv           ~6 KB
turnover.csv                ~3 KB
model_comparison.csv        ~1 KB

Each plot                   ~100-300 KB (PNG, 150 DPI)

Total per run:              ~1-2 MB
Total comparison run:       ~5-8 MB
```

## Reading Outputs in Python

```python
import pandas as pd
import json

# Load summary
with open('outputs/backtest_summary.json') as f:
    summary = json.load(f)
print(f"Sharpe: {summary['dynamic_factor_portfolio']['sharpe']:.2f}")

# Load returns
returns = pd.read_csv('outputs/monthly_returns.csv', index_col=0, parse_dates=True)
print(f"Avg turnover: {returns['turnover'].mean():.2%}")

# Load stock weights
weights = pd.read_csv('outputs/stock_weights.csv', index_col=0, parse_dates=True)
print(weights.head())

# Load comparison
comparison = pd.read_csv('outputs/model_comparison.csv', index_col=0)
print(comparison.sort_values('sharpe_ratio', ascending=False))
```

## Reading Outputs in R

```r
library(jsonlite)
library(readr)

# Load summary
summary <- fromJSON('outputs/backtest_summary.json')
cat(sprintf("Sharpe: %.2f\n", summary$dynamic_factor_portfolio$sharpe))

# Load returns
returns <- read_csv('outputs/monthly_returns.csv')
cat(sprintf("Avg turnover: %.2f%%\n", mean(returns$turnover) * 100))

# Load stock weights
weights <- read_csv('outputs/stock_weights.csv')
head(weights)

# Load comparison
comparison <- read_csv('outputs/model_comparison.csv')
comparison[order(-comparison$sharpe_ratio), ]
```

## Output Validation Checklist

After running the pipeline, verify:

✅ All CSV files exist and are non-empty
✅ backtest_summary.json contains all expected metrics
✅ monthly_returns.csv has ret_gross, ret_net, turnover columns
✅ stock_weights.csv has all 6 tickers as columns
✅ All plots generated without errors
✅ Sharpe ratio is reasonable (typically 0.5-1.5)
✅ OOS R² is positive (indicates predictive power)
✅ Average turnover is reasonable (typically 20-40%)
✅ No NaN values in critical columns

## Troubleshooting

**Missing files?**
→ Check logs for errors during backtest
→ Ensure minimum 24 months training data

**NaN in metrics?**
→ Check if returns series is too short
→ Verify data quality in monthly_panel.parquet

**Plots not generated?**
→ Check matplotlib backend
→ Verify outputs/ directory is writable

**Comparison CSV empty?**
→ Ensure --compare-models flag was used
→ Check logs for model-specific errors
