from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig, PipelineConfig
from .utils import get_logger


def _to_month_end(df: pd.DataFrame, price_column: str) -> pd.DataFrame:
    df = df.copy()
    df.set_index("date", inplace=True)
    out = (
        df.groupby("ticker")[price_column]
        .resample("M")
        .last()
        .rename("price")
        .to_frame()
        .reset_index()
    )
    out.rename(columns={"date": "month_end"}, inplace=True)
    return out


def _compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    monthly_prices = monthly_prices.sort_values(["ticker", "month_end"]).copy()
    monthly_prices["ret_1m"] = (
        monthly_prices.groupby("ticker")["price"].pct_change(fill_method=None)
    )
    return monthly_prices


def _momentum_features(
    monthly: pd.DataFrame, cfg: FeatureConfig
) -> pd.DataFrame:
    df = monthly.sort_values(["ticker", "month_end"]).copy()
    g = df.groupby("ticker")

    # Classic 12-1 and 6-1 momentum using shifted prices:
    # mom_12_1 = P_{t-1} / P_{t-12} - 1
    # mom_6_1  = P_{t-1} / P_{t-6}  - 1
    price = g["price"]
    price_lag1 = price.shift(1)
    price_lag12 = price.shift(cfg.mom_12_1_window)
    price_lag6 = price.shift(cfg.mom_6_1_window)

    df["mom_12_1"] = price_lag1 / price_lag12 - 1.0
    df["mom_6_1"] = price_lag1 / price_lag6 - 1.0

    df["rev_1m"] = -df["ret_1m"]
    return df


def _low_risk_features(
    daily: pd.DataFrame,
    spy: pd.DataFrame,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    from scipy.stats import linregress

    logger = get_logger()
    daily = daily.sort_values(["ticker", "date"]).copy()
    spy = spy.sort_values("date").copy()

    daily["ret_d"] = daily.groupby("ticker")["adj_close"].pct_change(fill_method=None)
    spy["ret_d"] = spy["adj_close"].pct_change(fill_method=None)

    merged = daily.merge(spy[["date", "ret_d"]], on="date", suffixes=("", "_mkt"))

    def rolling_beta(x: pd.DataFrame, window: int) -> pd.Series:
        """
        Rolling CAPM beta vs. market using daily returns.
        Uses scipy.stats.linregress; slope is the beta.
        """
        returns = x.set_index("date")[["ret_d", "ret_d_mkt"]].dropna()
        if returns.empty:
            return pd.Series(index=x.index, dtype=float)
        out = []
        idx = []
        for i in range(len(returns)):
            if i + 1 < window:
                out.append(np.nan)
            else:
                window_df = returns.iloc[i + 1 - window : i + 1]
                res = linregress(window_df["ret_d_mkt"], window_df["ret_d"])
                out.append(res.slope)
            idx.append(returns.index[i])
        s = pd.Series(out, index=idx)
        return x["date"].map(s)

    def realized_vol(x: pd.Series, window: int) -> pd.Series:
        return x.rolling(window=window, min_periods=window).std() * np.sqrt(252.0)

    def downside_vol(x: pd.Series, window: int) -> pd.Series:
        neg = x.clip(upper=0.0)
        return neg.rolling(window=window, min_periods=window).std() * np.sqrt(252.0)

    merged["beta_12m"] = (
        merged.groupby("ticker")
        .apply(lambda x: rolling_beta(x, cfg.low_risk_lookback_days_long))
        .reset_index(level=0, drop=True)
    )

    merged["vol_3m"] = merged.groupby("ticker")["ret_d"].transform(
        lambda x: realized_vol(x, cfg.low_risk_lookback_days_short)
    )
    merged["vol_12m"] = merged.groupby("ticker")["ret_d"].transform(
        lambda x: realized_vol(x, cfg.low_risk_lookback_days_long)
    )
    merged["down_vol_12m"] = merged.groupby("ticker")["ret_d"].transform(
        lambda x: downside_vol(x, cfg.low_risk_lookback_days_long)
    )

    daily_feat = merged[
        [
            "date",
            "ticker",
            "beta_12m",
            "vol_3m",
            "vol_12m",
            "down_vol_12m",
        ]
    ]

    # Group by ticker, resample to month-end, then drop duplicate ticker column
    monthly = (
        daily_feat.set_index("date")
        .groupby("ticker")
        .resample("M")
        .last()
    )
    if "ticker" in monthly.columns:
        monthly = monthly.drop(columns="ticker")
    monthly = monthly.reset_index()
    monthly.rename(columns={"date": "month_end"}, inplace=True)
    return monthly


def _liquidity_features(daily: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    daily = daily.sort_values(["ticker", "date"]).copy()
    daily["dollar_vol"] = daily["adj_close"] * daily["volume"]
    g = daily.groupby("ticker")

    daily["dollar_vol_1m"] = g["dollar_vol"].transform(
        lambda x: x.rolling(cfg.liquidity_lookback_days, min_periods=5).mean()
    )
    daily["turnover_proxy"] = g["volume"].transform(
        lambda x: x.rolling(cfg.liquidity_lookback_days, min_periods=5).mean()
    )
    daily["amihud"] = g.apply(
        lambda df: df["ret_d"].abs()
        .rolling(cfg.amihud_lookback_days, min_periods=5)
        .sum()
        / df["dollar_vol"].rolling(cfg.amihud_lookback_days, min_periods=5).sum()
    ).reset_index(level=0, drop=True)

    monthly = (
        daily.set_index("date")
        .groupby("ticker")
        .resample("M")
        .last()
    )
    if "ticker" in monthly.columns:
        monthly = monthly.drop(columns="ticker")
    monthly = monthly.reset_index()
    monthly.rename(columns={"date": "month_end"}, inplace=True)
    return monthly[
        [
            "month_end",
            "ticker",
            "dollar_vol_1m",
            "turnover_proxy",
            "amihud",
        ]
    ]


def _behavioural_features(daily: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    daily = daily.sort_values(["ticker", "date"]).copy()
    g = daily.groupby("ticker")

    daily["ret_5d"] = g["adj_close"].transform(
        lambda x: x.pct_change(periods=cfg.overreaction_lookback_days, fill_method=None)
    )
    daily["vol_mean_60"] = g["volume"].transform(
        lambda x: x.rolling(cfg.turnover_spike_lookback_days, min_periods=10).mean()
    )

    def zscore(x: pd.Series) -> pd.Series:
        return (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else x * 0.0

    daily["z_ret_5d"] = g["ret_5d"].transform(zscore)
    daily["z_abn_volume"] = g.apply(
        lambda df: zscore(df["volume"] / df["vol_mean_60"] - 1.0)
    ).reset_index(level=0, drop=True)
    daily["overreaction_score"] = daily["z_ret_5d"] * daily["z_abn_volume"]

    rolling_max = g["adj_close"].transform(
        lambda x: x.rolling(cfg.window_52w_days, min_periods=20).max()
    )
    rolling_min = g["adj_close"].transform(
        lambda x: x.rolling(cfg.window_52w_days, min_periods=20).min()
    )
    daily["dist_to_52w_high"] = daily["adj_close"] / rolling_max - 1.0
    daily["dist_to_52w_low"] = daily["adj_close"] / rolling_min - 1.0

    daily["turnover_spike"] = daily["volume"] / daily["vol_mean_60"] - 1.0
    daily["disposition_high"] = (
        daily["turnover_spike"] * (daily["dist_to_52w_high"] > -0.05).astype(float)
    )
    daily["disposition_low"] = (
        daily["turnover_spike"] * (daily["dist_to_52w_low"] < 0.05).astype(float)
    )

    daily["attention_proxy"] = (
        daily.groupby("ticker")["volume"].transform(zscore).fillna(0.0)
    )

    monthly = (
        daily.set_index("date")
        .groupby("ticker")
        .resample("M")
        .last()
    )
    if "ticker" in monthly.columns:
        monthly = monthly.drop(columns="ticker")
    monthly = monthly.reset_index()
    monthly.rename(columns={"date": "month_end"}, inplace=True)
    return monthly[
        [
            "month_end",
            "ticker",
            "overreaction_score",
            "disposition_high",
            "disposition_low",
            "attention_proxy",
        ]
    ]


def build_monthly_feature_panel(
    cfg: PipelineConfig,
    daily_prices: pd.DataFrame,
    spy_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a single monthly feature panel indexed by (month_end, ticker).
    """
    logger = get_logger()
    logger.info("Building monthly feature panel")

    daily = daily_prices.sort_values(["ticker", "date"]).copy()
    spy = spy_daily.sort_values("date").copy()

    daily["ret_d"] = daily.groupby("ticker")["adj_close"].pct_change(fill_method=None)

    monthly_prices = _to_month_end(daily, price_column="adj_close")
    monthly_prices = _compute_monthly_returns(monthly_prices)

    mom_df = _momentum_features(monthly_prices, cfg.features)
    low_risk_df = _low_risk_features(daily, spy, cfg.features)
    liq_df = _liquidity_features(daily, cfg.features)
    beh_df = _behavioural_features(daily, cfg.features)

    merged = (
        mom_df.merge(low_risk_df, on=["month_end", "ticker"], how="left")
        .merge(liq_df, on=["month_end", "ticker"], how="left")
        .merge(beh_df, on=["month_end", "ticker"], how="left")
    )

    merged = merged[merged["month_end"] >= pd.to_datetime("2010-01-31")]
    return merged


