from pathlib import Path

path = Path('src/factor_portfolios.py')
text = path.read_text(encoding='utf-8')
old = '''        def _per_month(g: pd.DataFrame) -> pd.DataFrame:
            if g.empty:
                return g[["month_end", "ticker"]].assign(weight=0.0)
            if factor_name == "lowvol":
                raw_signal = -_zscore_cross_section(g[col])
            else:
                raw_signal = _zscore_cross_section(g[col])
            weights = _normalize_weights(raw_signal)
            return pd.DataFrame(
                {
                    "month_end": g["month_end"].values,
                    "ticker": g["ticker"].values,
                    "weight": weights.values,
                }
            )
'''
new = '''        def _per_month(g: pd.DataFrame) -> pd.DataFrame:
            month_end = g.name
            if g.empty:
                return pd.DataFrame(columns=["month_end", "ticker", "weight"])
            if factor_name == "lowvol":
                raw_signal = -_zscore_cross_section(g[col])
            else:
                raw_signal = _zscore_cross_section(g[col])
            weights = _normalize_weights(raw_signal)
            return pd.DataFrame(
                {
                    "month_end": [month_end] * len(g),
                    "ticker": g["ticker"].values,
                    "weight": weights.values,
                }
            )
'''
if old not in text:
    raise SystemExit('Target block not found')
path.write_text(text.replace(old, new), encoding='utf-8')
print('Updated src/factor_portfolios.py')
