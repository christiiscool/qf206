from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import get_logger



def _zscore_cross_section(s: pd.Series) -> pd.Series:
    if s.std(ddof=0) == 0 or s.isna().all():
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)



def _normalize_weights(signal: pd.Series) -> pd.Series:
    """
    Cross-sectional normalization: w_i = signal_i / sum_j |signal_j|.
    Falls back to equal weights if all signals are zero/NaN.
    """
    s = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = np.abs(s).sum()
    if denom <= 0:
        n = len(s)
        return pd.Series(1.0 / n, index=s.index)
    return s / denom



def build_factor_signals(
    cfg: PipelineConfig, panel_6: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Build factor-specific stock weights for the 6-stock universe.

    For each month and stock:
      - momentum:     zscore(mom_12_1)
      - reversal:     zscore(rev_1m)
      - lowvol:       -zscore(vol_12m)
      - behavioural:  zscore(overreaction_score)

    Returns a dict mapping factor name -> DataFrame[month_end, ticker, weight].
    """
    logger = get_logger()
    df = panel_6.copy()
    df = df[df["ticker"].isin(cfg.demo_tickers)]

    factors = {
        "momentum": "mom_12_1",
        "reversal": "rev_1m",
        "lowvol": "vol_12m",
        "behavioural": "overreaction_score",
    }

    out: Dict[str, pd.DataFrame] = {}

    for factor_name, col in factors.items():
        if col not in df.columns:
            raise KeyError(f"Required factor column '{col}' missing from panel")

        def _per_month(g: pd.DataFrame) -> pd.DataFrame:
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

        weights_df = (
            df.groupby("month_end", group_keys=False)
            .apply(_per_month)
            .reset_index(drop=True)
        )
        out[factor_name] = weights_df

    logger.info("Built factor signals for factors: %s", ", ".join(out.keys()))
    return out



def compute_factor_returns(
    panel_6: pd.DataFrame, factor_weights: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute factor returns as weighted sums of forward 1M stock returns.

    r_factor_t = sum_i w_i,t * forward_1m_return_i,t
    """
    df = panel_6[["month_end", "ticker", "forward_1m_return"]].copy()
    df = df.dropna(subset=["forward_1m_return"])

    factor_ret_frames = []
    for factor_name, w_df in factor_weights.items():
        merged = df.merge(
            w_df, on=["month_end", "ticker"], how="inner", suffixes=("", "_w")
        )
        merged["contrib"] = merged["weight"] * merged["forward_1m_return"]
        fac = (
            merged.groupby("month_end")["contrib"]
            .sum()
            .rename(factor_name)
            .to_frame()
        )
        factor_ret_frames.append(fac)

    if not factor_ret_frames:
        raise ValueError("No factor weights provided to compute_factor_returns.")

    factor_returns = pd.concat(factor_ret_frames, axis=1).sort_index()
    return factor_returns
