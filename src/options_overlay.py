from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import OptionsRiskConfig
from .utils import get_logger


def _select_put_contract(
    month_slice: pd.DataFrame,
    cfg: OptionsRiskConfig,
    fallback_spot: float | None = None,
) -> pd.Series | None:
    puts = month_slice[month_slice["cp_flag"] == "P"].dropna(
        subset=["mid", "delta", "dte", "strike_price"]
    ).copy()
    puts = puts[puts["mid"] > 0].copy()
    if puts.empty:
        return None

    if "underlying_price" not in puts.columns:
        puts["underlying_price"] = np.nan
    if fallback_spot is not None and np.isfinite(fallback_spot):
        puts["underlying_price"] = puts["underlying_price"].fillna(float(fallback_spot))

    puts["delta_distance"] = (puts["delta"].abs() - cfg.put_target_delta_abs).abs()
    puts["dte_distance"] = (puts["dte"] - cfg.put_target_dte_days).abs()
    puts["otm_gap"] = np.where(
        puts["underlying_price"].notna() & (puts["underlying_price"] > 0),
        (puts["strike_price"] / puts["underlying_price"] - 1.0).abs(),
        np.inf,
    )

    chosen = puts.sort_values(
        ["delta_distance", "dte_distance", "open_interest", "volume", "otm_gap"],
        ascending=[True, True, False, False, True],
    ).iloc[0]
    return chosen


def _spy_close_on_or_before(spy_close: pd.Series, date: pd.Timestamp) -> float:
    available = spy_close.loc[spy_close.index <= date]
    if available.empty:
        return np.nan
    return float(available.iloc[-1])


def _intrinsic_put_value(strike: float, spot: float) -> float:
    if not np.isfinite(strike) or not np.isfinite(spot):
        return np.nan
    return float(max(strike - spot, 0.0))


def _build_trade_path(
    option_quotes: pd.Series,
    entry_date: pd.Timestamp,
    entry_mid: float,
    exdate: pd.Timestamp,
    strike: float,
    holding_dates: pd.DatetimeIndex,
    spy_close: pd.Series,
) -> Tuple[pd.Series, pd.Timestamp, float, str]:
    if holding_dates.empty:
        return pd.Series(dtype=float), entry_date, entry_mid, "no_holding_window"

    dates = pd.DatetimeIndex([entry_date]).append(holding_dates)
    marks = option_quotes.reindex(dates).ffill()
    marks.loc[entry_date] = entry_mid

    target_end = holding_dates.max()
    settlement_date = min(exdate, target_end)
    settlement_marks = option_quotes.loc[option_quotes.index <= settlement_date]

    if not settlement_marks.empty:
        exit_value = float(settlement_marks.iloc[-1])
        exit_reason = "quote_mark"
        exit_date = settlement_marks.index[-1]
    else:
        exit_spot = _spy_close_on_or_before(spy_close, settlement_date)
        exit_value = _intrinsic_put_value(strike, exit_spot)
        exit_reason = "intrinsic_fallback"
        exit_date = settlement_date

    if exdate <= target_end:
        post_expiry_mask = marks.index >= exdate
        marks.loc[post_expiry_mask] = exit_value
    else:
        marks = marks.ffill()
        marks.loc[marks.index > exit_date] = exit_value

    marks = marks.ffill().fillna(entry_mid)
    marks = marks.clip(lower=0.0)
    return marks, pd.Timestamp(exit_date), float(exit_value), exit_reason


def build_monthly_put_hedge_book(
    options_df: pd.DataFrame,
    spy_daily: pd.DataFrame,
    cfg: OptionsRiskConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_logger()

    if options_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = options_df.copy().sort_values(["date", "exdate", "optionid"])
    df["month_end"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

    snapshot_dates = df.groupby("month_end")["date"].max().rename("snapshot_date")
    snapshot = df.merge(snapshot_dates, on="month_end", how="inner")
    snapshot = snapshot[snapshot["date"] == snapshot["snapshot_date"]].copy()
    snapshot = snapshot[(snapshot["dte"] >= cfg.min_dte) & (snapshot["dte"] <= cfg.max_dte)].copy()

    spy = spy_daily.copy().sort_values("date")
    spy["date"] = pd.to_datetime(spy["date"])
    spy_close = spy.set_index("date")["adj_close"].sort_index()

    months = sorted(snapshot["month_end"].dropna().unique())
    trade_records = []
    daily_records = []

    for idx, month_end in enumerate(months):
        if idx + 1 >= len(months):
            break

        next_month_end = pd.Timestamp(months[idx + 1])
        month_slice = snapshot[snapshot["month_end"] == month_end]
        fallback_spot = _spy_close_on_or_before(spy_close, pd.Timestamp(month_end))
        chosen = _select_put_contract(month_slice, cfg, fallback_spot=fallback_spot)
        if chosen is None:
            continue

        entry_date = pd.Timestamp(chosen["date"])
        exdate = pd.Timestamp(chosen["exdate"])
        optionid = chosen.get("optionid", np.nan)
        entry_mid = float(chosen["mid"])
        strike = float(chosen["strike_price"])
        underlying_entry = (
            float(chosen["underlying_price"])
            if pd.notna(chosen.get("underlying_price", np.nan))
            else float(fallback_spot)
        )

        holding_dates = spy_close.index[(spy_close.index > pd.Timestamp(month_end)) & (spy_close.index <= next_month_end)]
        if len(holding_dates) == 0:
            continue

        option_quotes = (
            df[df["optionid"] == optionid]
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")["mid"]
        )

        marks, exit_date, exit_value, exit_reason = _build_trade_path(
            option_quotes=option_quotes,
            entry_date=entry_date,
            entry_mid=entry_mid,
            exdate=exdate,
            strike=strike,
            holding_dates=holding_dates,
            spy_close=spy_close,
        )
        if marks.empty or not np.isfinite(entry_mid) or entry_mid <= 0:
            continue

        normalized_value = marks / entry_mid
        daily_unit_pnl = normalized_value.diff().dropna()
        total_return = float(normalized_value.iloc[-1] - 1.0)
        underlying_exit = _spy_close_on_or_before(spy_close, exit_date)

        trade_records.append(
            {
                "month_end": pd.Timestamp(month_end),
                "signal_date": entry_date,
                "next_month_end": next_month_end,
                "optionid": optionid,
                "exdate": exdate,
                "strike_price": strike,
                "entry_mid": entry_mid,
                "exit_date": exit_date,
                "exit_value": exit_value,
                "underlying_entry": underlying_entry,
                "underlying_exit": underlying_exit,
                "option_unit_total_return": total_return,
                "exit_reason": exit_reason,
            }
        )

        for date, unit_pnl in daily_unit_pnl.items():
            daily_records.append(
                {
                    "month_end": pd.Timestamp(month_end),
                    "date": pd.Timestamp(date),
                    "option_unit_pnl": float(unit_pnl),
                }
            )

    if trade_records:
        tradebook = pd.DataFrame(trade_records).sort_values("month_end").reset_index(drop=True)
    else:
        tradebook = pd.DataFrame(
            columns=[
                "month_end",
                "signal_date",
                "next_month_end",
                "optionid",
                "exdate",
                "strike_price",
                "entry_mid",
                "exit_date",
                "exit_value",
                "underlying_entry",
                "underlying_exit",
                "option_unit_total_return",
                "exit_reason",
            ]
        )
    if daily_records:
        daily_book = pd.DataFrame(daily_records).sort_values(["date", "month_end"]).reset_index(drop=True)
    else:
        daily_book = pd.DataFrame(columns=["month_end", "date", "option_unit_pnl"])
    logger.info("Built monthly SPY put hedge book for %d months", len(tradebook))
    return tradebook, daily_book
