from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OverlayConfig, PipelineConfig
from .utils import get_logger


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(252.0)


def _max_drawdown(series: pd.Series) -> pd.Series:
    cum = (1.0 + series).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return dd


def apply_options_overlay(
    cfg: PipelineConfig,
    portfolio_returns: pd.Series,
    spy_daily: pd.DataFrame,
) -> pd.Series:
    """
    Apply a simple put-spread overlay triggered by SPY risk regime.
    """
    logger = get_logger()
    if not cfg.overlay.enabled:
        return portfolio_returns

    spy = spy_daily.sort_values("date").copy()
    spy["ret_d"] = spy["adj_close"].pct_change()
    spy["realized_vol"] = _realized_vol(
        spy["ret_d"], cfg.overlay.realized_vol_lookback_days
    )
    spy["drawdown"] = _max_drawdown(spy["ret_d"])

    vol_mean = spy["realized_vol"].mean()
    vol_std = spy["realized_vol"].std(ddof=0)
    if vol_std == 0 or np.isnan(vol_std):
        spy["vol_z"] = 0.0
    else:
        spy["vol_z"] = (spy["realized_vol"] - vol_mean) / vol_std

    spy["month_end"] = spy["date"].values.astype("datetime64[M]") + np.timedelta64(
        0, "D"
    )
    spy_m = (
        spy.groupby("month_end")[["vol_z", "drawdown"]]
        .last()
        .reset_index()
        .set_index("month_end")
    )

    port = portfolio_returns.copy().to_frame("ret")
    port.index = pd.to_datetime(port.index)
    spy_m = spy_m.reindex(port.index).ffill()

    triggered = (
        (spy_m["vol_z"] >= cfg.overlay.vol_zscore_trigger)
        | (spy_m["drawdown"].abs() >= cfg.overlay.max_drawdown_trigger)
    ).astype(int)

    prem = cfg.overlay.monthly_hedge_premium_bps / 10000.0
    port["hedge_premium"] = -prem * triggered

    hedge_payoff = np.where(
        port["ret"] < cfg.overlay.put_spread_cap,
        np.clip(
            port["ret"] - cfg.overlay.put_spread_cap,
            cfg.overlay.put_spread_floor,
            0.0,
        ),
        0.0,
    )
    port["hedge_payoff"] = hedge_payoff * triggered

    adjusted = port["ret"] + port["hedge_premium"] + port["hedge_payoff"]
    logger.info("Applied options overlay on %d months", len(adjusted))
    return adjusted

