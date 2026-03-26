from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import OptionsRiskConfig
from .options_signal_data import load_wrds_options_export
from .utils import get_logger


REQUIRED_SIGNAL_COLUMNS = [
    "atm_iv",
    "otm_put_iv",
    "skew",
    "put_call_volume_ratio",
    "put_call_open_interest_ratio",
    "deep_otm_put_demand",
]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _pick_atm_iv(snapshot: pd.DataFrame) -> float:
    candidates = snapshot.dropna(subset=["impl_volatility", "delta"]).copy()
    if candidates.empty:
        return np.nan
    candidates["atm_distance"] = (candidates["delta"].abs() - 0.50).abs()
    return float(candidates.sort_values(["atm_distance", "dte", "open_interest"], ascending=[True, True, False]).iloc[0]["impl_volatility"])


def _pick_otm_put_iv(snapshot: pd.DataFrame) -> float:
    puts = snapshot[snapshot["cp_flag"] == "P"].dropna(subset=["impl_volatility", "delta"]).copy()
    if puts.empty:
        return np.nan
    puts["abs_delta"] = puts["delta"].abs()
    band = puts[(puts["abs_delta"] >= 0.20) & (puts["abs_delta"] <= 0.30)]
    if band.empty:
        puts["target_distance"] = (puts["abs_delta"] - 0.25).abs()
        band = puts.sort_values(["target_distance", "dte", "open_interest"], ascending=[True, True, False]).head(3)
    return float(band["impl_volatility"].median())


def _deep_otm_put_proxy(snapshot: pd.DataFrame) -> float:
    puts = snapshot[snapshot["cp_flag"] == "P"].copy()
    if puts.empty:
        return np.nan

    deep_puts = puts.dropna(subset=["delta"]).copy()
    if deep_puts.empty:
        return np.nan

    deep_puts = deep_puts[deep_puts["delta"].abs() < 0.15]
    if deep_puts.empty:
        return np.nan

    vol_share = _safe_ratio(deep_puts["volume"].sum(), puts["volume"].sum())
    oi_share = _safe_ratio(deep_puts["open_interest"].sum(), puts["open_interest"].sum())

    available = [val for val in [vol_share, oi_share] if pd.notna(val)]
    if not available:
        return np.nan
    return float(np.mean(available))


def build_monthly_options_indicators(options_df: pd.DataFrame, cfg: OptionsRiskConfig) -> pd.DataFrame:
    logger = get_logger()
    df = options_df.copy()
    df["month_end"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

    last_dates = df.groupby("month_end")["date"].max().rename("snapshot_date")
    snapshot = df.merge(last_dates, on="month_end", how="inner")
    snapshot = snapshot[snapshot["date"] == snapshot["snapshot_date"]].copy()
    snapshot = snapshot[(snapshot["dte"] >= cfg.min_dte) & (snapshot["dte"] <= cfg.max_dte)].copy()

    monthly_records = []
    for month_end, month_slice in snapshot.groupby("month_end"):
        puts = month_slice[month_slice["cp_flag"] == "P"].copy()
        calls = month_slice[month_slice["cp_flag"] == "C"].copy()

        atm_iv = _pick_atm_iv(month_slice)
        otm_put_iv = _pick_otm_put_iv(month_slice)
        skew = otm_put_iv - atm_iv if pd.notna(atm_iv) and pd.notna(otm_put_iv) else np.nan

        record = {
            "month_end": month_end,
            "signal_date": month_slice["date"].max(),
            "atm_iv": atm_iv,
            "otm_put_iv": otm_put_iv,
            "skew": skew,
            "put_call_volume_ratio": _safe_ratio(puts["volume"].sum(), calls["volume"].sum()),
            "put_call_open_interest_ratio": _safe_ratio(puts["open_interest"].sum(), calls["open_interest"].sum()),
            "deep_otm_put_demand": _deep_otm_put_proxy(month_slice),
            "contracts_used": int(len(month_slice)),
            "put_contracts": int(len(puts)),
            "call_contracts": int(len(calls)),
            "underlying_price": float(month_slice["underlying_price"].dropna().median()) if month_slice["underlying_price"].notna().any() else np.nan,
        }
        monthly_records.append(record)

    monthly = pd.DataFrame(monthly_records).sort_values("month_end").reset_index(drop=True)
    logger.info("Built monthly options indicators for %d months", len(monthly))
    return monthly


def _rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    trailing_mean = series.shift(1).rolling(window=lookback, min_periods=max(12, lookback // 2)).mean()
    trailing_std = series.shift(1).rolling(window=lookback, min_periods=max(12, lookback // 2)).std(ddof=0)
    zscore = (series - trailing_mean) / trailing_std.replace(0.0, np.nan)
    return zscore


def add_warning_score(monthly_indicators: pd.DataFrame, cfg: OptionsRiskConfig) -> pd.DataFrame:
    monthly = monthly_indicators.copy().sort_values("month_end").reset_index(drop=True)

    z_map = {
        "skew_z": _rolling_zscore(monthly["skew"], cfg.rolling_lookback_months),
        "put_call_volume_ratio_z": _rolling_zscore(monthly["put_call_volume_ratio"], cfg.rolling_lookback_months),
        "deep_otm_put_demand_z": _rolling_zscore(monthly["deep_otm_put_demand"], cfg.rolling_lookback_months),
    }
    for col, values in z_map.items():
        monthly[col] = values

    z_cols = list(z_map.keys())
    monthly["warning_score"] = monthly[z_cols].mean(axis=1, skipna=True)

    if cfg.use_rolling_percentile_threshold:
        rolling_threshold = (
            monthly["warning_score"]
            .shift(1)
            .rolling(window=cfg.rolling_lookback_months, min_periods=max(12, cfg.rolling_lookback_months // 2))
            .quantile(cfg.warning_percentile)
        )
        monthly["warning_threshold"] = rolling_threshold
        monthly["warning_flag"] = (monthly["warning_score"] > monthly["warning_threshold"]).astype(int)
    else:
        monthly["warning_threshold"] = cfg.warning_z_threshold
        monthly["warning_flag"] = (monthly["warning_score"] > cfg.warning_z_threshold).astype(int)

    monthly["score_mild_threshold"] = (
        monthly["warning_score"]
        .shift(1)
        .rolling(window=cfg.rolling_lookback_months, min_periods=max(12, cfg.rolling_lookback_months // 2))
        .quantile(cfg.score_scaled_mild_percentile)
    )
    monthly["score_high_threshold"] = (
        monthly["warning_score"]
        .shift(1)
        .rolling(window=cfg.rolling_lookback_months, min_periods=max(12, cfg.rolling_lookback_months // 2))
        .quantile(cfg.score_scaled_high_percentile)
    )
    monthly["score_extreme_threshold"] = (
        monthly["warning_score"]
        .shift(1)
        .rolling(window=cfg.rolling_lookback_months, min_periods=max(12, cfg.rolling_lookback_months // 2))
        .quantile(cfg.score_scaled_extreme_percentile)
    )
    monthly["fixed_hedge_budget"] = np.where(monthly["warning_flag"] == 1, cfg.fixed_hedge_budget_pct_nav, 0.0)
    monthly["fixed_overlay_equity_allocation"] = 1.0 - monthly["fixed_hedge_budget"]
    monthly["score_scaled_equity_allocation"] = 1.0
    monthly["score_scaled_hedge_budget"] = 0.0
    return monthly


def add_market_stress_confirmation(
    warning_signals: pd.DataFrame,
    spy_daily: pd.DataFrame,
    cfg: OptionsRiskConfig,
) -> pd.DataFrame:
    monthly = warning_signals.copy().sort_values("month_end").reset_index(drop=True)
    if monthly.empty or spy_daily.empty:
        return monthly

    spy = spy_daily.sort_values("date").copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy["ret_d"] = spy["adj_close"].pct_change(fill_method=None)
    spy["realized_vol_3m"] = spy["ret_d"].rolling(window=63, min_periods=40).std() * np.sqrt(252.0)

    equity_curve = (1.0 + spy["ret_d"].fillna(0.0)).cumprod()
    rolling_peak = equity_curve.cummax()
    spy["drawdown"] = equity_curve / rolling_peak - 1.0

    monthly_market = (
        spy.set_index("date")[["realized_vol_3m", "drawdown"]]
        .resample("M")
        .last()
        .reset_index()
        .rename(columns={"date": "month_end"})
    )
    monthly_market["vol_threshold"] = (
        monthly_market["realized_vol_3m"]
        .shift(1)
        .rolling(window=cfg.rolling_lookback_months, min_periods=max(12, cfg.rolling_lookback_months // 2))
        .quantile(cfg.market_vol_percentile)
    )
    monthly_market["market_vol_flag"] = (
        monthly_market["realized_vol_3m"] > monthly_market["vol_threshold"]
    ).fillna(False).astype(int)
    monthly_market["market_drawdown_flag"] = (
        monthly_market["drawdown"] < cfg.market_drawdown_trigger
    ).fillna(False).astype(int)
    monthly_market["market_stress_flag"] = (
        (monthly_market["market_vol_flag"] == 1) | (monthly_market["market_drawdown_flag"] == 1)
    ).astype(int)

    monthly = monthly.merge(
        monthly_market[
            [
                "month_end",
                "realized_vol_3m",
                "drawdown",
                "vol_threshold",
                "market_vol_flag",
                "market_drawdown_flag",
                "market_stress_flag",
            ]
        ],
        on="month_end",
        how="left",
    )

    monthly["options_warning_flag"] = monthly["warning_flag"].fillna(0).astype(int)
    if cfg.require_market_confirmation:
        monthly["warning_flag"] = (
            (monthly["options_warning_flag"] == 1) & (monthly["market_stress_flag"].fillna(0).astype(int) == 1)
        ).astype(int)

    monthly["fixed_hedge_budget"] = np.where(monthly["warning_flag"] == 1, cfg.fixed_hedge_budget_pct_nav, 0.0)
    monthly["fixed_overlay_equity_allocation"] = 1.0 - monthly["fixed_hedge_budget"]
    confirmed_stress = monthly["market_stress_flag"].fillna(0).astype(int) == 1
    mild = confirmed_stress & (monthly["warning_score"] >= monthly["score_mild_threshold"])
    high = confirmed_stress & (monthly["warning_score"] >= monthly["score_high_threshold"])
    extreme = confirmed_stress & (monthly["warning_score"] >= monthly["score_extreme_threshold"])

    monthly["score_scaled_hedge_budget"] = 0.0
    monthly.loc[mild, "score_scaled_hedge_budget"] = cfg.score_scaled_mild_budget_pct_nav
    monthly.loc[high, "score_scaled_hedge_budget"] = cfg.score_scaled_high_budget_pct_nav
    monthly.loc[extreme, "score_scaled_hedge_budget"] = cfg.score_scaled_extreme_budget_pct_nav
    monthly["score_scaled_equity_allocation"] = 1.0 - monthly["score_scaled_hedge_budget"]
    monthly["score_scaled_flag"] = (monthly["score_scaled_hedge_budget"] > 0.0).astype(int)
    return monthly


def build_options_warning_signals(path: Path | str, cfg: OptionsRiskConfig) -> pd.DataFrame:
    options_df = load_wrds_options_export(path, cfg)
    monthly_indicators = build_monthly_options_indicators(options_df, cfg)
    return add_warning_score(monthly_indicators, cfg)
