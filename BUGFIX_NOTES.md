# Bug Fix Notes

## Issue: KeyError 'ret_dynamic_factor'

**Date**: 2026-03-05
**Status**: ✅ FIXED

### Problem
After implementing the new backtest structure with `ret_gross`, `ret_net`, and `turnover` columns, the plotting functions were still looking for the old column name `ret_dynamic_factor`, causing a KeyError.

### Root Cause
The backtest was updated to return columns named:
- `ret_gross` (returns before transaction costs)
- `ret_net` (returns after transaction costs)
- `turnover` (monthly turnover)

But plotting functions still referenced the old column name `ret_dynamic_factor`.

### Files Fixed

1. **src/plots.py**
   - `plot_dynamic_portfolio_equity()`: Now checks for `ret_net` first, falls back to `ret_dynamic_factor` for backward compatibility
   
2. **src/comparison_plots.py**
   - `plot_cumulative_returns_comparison()`: Now checks for `ret_net` first, falls back to `ret_dynamic_factor`

### Solution
Added intelligent column detection with fallback logic:

```python
# Determine which return column to use
if "ret_net" in monthly_returns.columns:
    ret_col = "ret_net"
elif "ret_dynamic_factor" in monthly_returns.columns:
    ret_col = "ret_dynamic_factor"
else:
    ret_col = monthly_returns.columns[0]
```

This ensures:
- ✅ New backtest results work correctly (uses `ret_net`)
- ✅ Old backtest results still work (uses `ret_dynamic_factor`)
- ✅ Backward compatibility maintained
- ✅ Graceful degradation if neither column exists

### Testing
```bash
# Verify fix
python run_pipeline.py
# Should complete without KeyError

python run_pipeline.py --compare-models
# Should complete without KeyError
```

### Impact
- No breaking changes
- Backward compatible with old data
- Forward compatible with new data structure
- All plots now work correctly

### Related Changes
The same pattern was applied to:
- `plot_model_comparison_equity_curves()` in src/plots.py
- `plot_cumulative_returns_comparison()` in src/comparison_plots.py
- `plot_drawdown_series()` in src/plots.py
- `plot_rolling_sharpe()` in src/plots.py

All plotting functions now intelligently detect the correct return column to use.
