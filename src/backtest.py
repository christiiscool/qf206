from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .factor_portfolios import build_factor_signals, compute_factor_returns
from .factor_timing_model import (
    build_factor_timing_dataset,
    build_regime_features,
    predict_factor_returns,
    train_factor_timing_model,
)
from .options_warning_signals import add_market_stress_confirmation, build_options_warning_signals
from .portfolio_allocator import combine_factor_and_stock_weights, compute_factor_allocations
from .utils import get_logger


@dataclass
class BacktestResult:
    monthly_returns: pd.DataFrame
    daily_returns: pd.DataFrame
    factor_allocations: pd.DataFrame
    stock_weights: pd.DataFrame
    adjusted_stock_weights: pd.DataFrame
    scenario_stock_weights: Dict[str, pd.DataFrame]
    prediction_metrics: Dict[str, float]
    turnover_series: pd.Series
    overlay_turnover_series: pd.Series
    options_signals: pd.DataFrame


def _compute_vol_scalar(history: list[float], cfg: PipelineConfig) -> float:
    if not cfg.factor_timing.enable_vol_targeting:
        return 1.0
    if len(history) < cfg.factor_timing.vol_lookback:
        return 1.0

    realized_vol = np.std(history[-cfg.factor_timing.vol_lookback :], ddof=1) * np.sqrt(12.0)
    if not np.isfinite(realized_vol) or realized_vol <= 0:
        return 1.0

    vol_scalar = cfg.factor_timing.target_vol / realized_vol
    return float(np.clip(vol_scalar, 0.5, 2.0))


def _load_options_signals(cfg: PipelineConfig, spy_daily: pd.DataFrame) -> pd.DataFrame:
    logger = get_logger()
    if not cfg.options_risk.enabled:
        return pd.DataFrame(columns=["month_end", "warning_flag", "warning_score", "lambda_t"])
    if cfg.options_risk.options_data_path is None:
        raise ValueError("Options risk overlay enabled but no options_data_path is configured.")

    options_signals = build_options_warning_signals(Path(cfg.options_risk.options_data_path), cfg.options_risk)
    options_signals = add_market_stress_confirmation(options_signals, spy_daily, cfg.options_risk)
    logger.info("Loaded options warning signals with %d monthly observations", len(options_signals))
    return options_signals


def _overlay_scenarios(cfg: PipelineConfig) -> Dict[str, float]:
    scenarios = {
        "under_hedged": float(cfg.options_risk.under_hedge_haircut),
        "fixed_overlay": float(cfg.options_risk.warning_haircut),
        "over_hedged": float(cfg.options_risk.over_hedge_haircut),
    }

    for haircut in cfg.options_risk.hedge_sweep_haircuts:
        label = f"hedge_{int(round(float(haircut) * 100)):02d}"
        scenarios[label] = float(haircut)

    return dict(sorted(scenarios.items(), key=lambda item: item[1], reverse=True))


def _compute_vol_benchmark_lambda(base_history: list[float], cfg: PipelineConfig) -> float:
    lookback = cfg.options_risk.vol_benchmark_lookback_months
    if len(base_history) < lookback:
        return 1.0

    realized_vol = np.std(base_history[-lookback:], ddof=1) * np.sqrt(12.0)
    if not np.isfinite(realized_vol) or realized_vol <= 0:
        return 1.0

    raw_lambda = cfg.options_risk.vol_benchmark_target_vol / realized_vol
    return float(np.clip(raw_lambda, cfg.options_risk.vol_benchmark_min_lambda, 1.0))


def _build_daily_scenario_returns(
    cfg: PipelineConfig,
    monthly_returns: pd.DataFrame,
    daily_prices: Optional[pd.DataFrame],
    scenario_stock_weights: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if daily_prices is None or daily_prices.empty:
        return pd.DataFrame()

    price_col = "adj_close" if "adj_close" in daily_prices.columns else "close"
    if price_col not in daily_prices.columns:
        return pd.DataFrame()

    prices = daily_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    price_matrix = (
        prices.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last")
        .sort_index()
    )
    asset_daily_returns = price_matrix.pct_change().fillna(0.0)

    monthly_index = list(monthly_returns.index)
    scenario_daily: Dict[str, pd.Series] = {}
    tc_rate = cfg.backtest.transaction_cost_bps / 10000.0

    for scenario_name, weights_df in scenario_stock_weights.items():
        if weights_df.empty:
            continue

        daily_parts = []
        turnover_col = "turnover" if scenario_name == "unhedged" else f"{scenario_name}_turnover"
        vol_scalar_col = "base_vol_scalar" if scenario_name == "unhedged" else f"{scenario_name}_vol_scalar"
        regime_cost_col = None if scenario_name == "unhedged" else f"{scenario_name}_regime_switch_cost"

        for idx, month in enumerate(monthly_index):
            if month not in weights_df.index:
                continue

            next_month = monthly_index[idx + 1] if idx + 1 < len(monthly_index) else None
            mask = asset_daily_returns.index > month
            if next_month is not None:
                mask &= asset_daily_returns.index <= next_month

            period_returns = asset_daily_returns.loc[mask]
            if period_returns.empty:
                continue

            weights = weights_df.loc[month].dropna()
            daily_portfolio = (
                period_returns.reindex(columns=weights.index, fill_value=0.0)
                .mul(weights, axis=1)
                .sum(axis=1)
            )

            total_cost = float(monthly_returns.at[month, turnover_col]) * tc_rate
            if regime_cost_col is not None and regime_cost_col in monthly_returns.columns:
                total_cost += float(monthly_returns.at[month, regime_cost_col])

            if total_cost != 0.0:
                daily_portfolio.iloc[0] = daily_portfolio.iloc[0] - total_cost

            if vol_scalar_col in monthly_returns.columns:
                daily_portfolio = daily_portfolio * float(monthly_returns.at[month, vol_scalar_col])

            daily_parts.append(daily_portfolio)

        if daily_parts:
            scenario_daily[scenario_name] = pd.concat(daily_parts).sort_index()

    if not scenario_daily:
        return pd.DataFrame()

    daily_df = pd.DataFrame(scenario_daily).sort_index()
    daily_df.index.name = "date"
    return daily_df


def run_walk_forward_backtest(
    cfg: PipelineConfig,
    panel: pd.DataFrame,
    daily_prices: Optional[pd.DataFrame],
    spy_daily: pd.DataFrame,
) -> BacktestResult:
    """
    Walk-forward backtest for dynamic factor allocation on the 6-stock universe.

    The options signal acts only as a regime-aware risk overlay by scaling the
    equity portfolio exposure through lambda_t.
    """
    logger = get_logger()

    panel_6 = panel[panel["ticker"].isin(cfg.demo_tickers)].copy()
    panel_6 = panel_6.sort_values(["month_end", "ticker"])

    factor_stock_weights = build_factor_signals(cfg, panel_6)
    factor_returns = compute_factor_returns(panel_6, factor_stock_weights)

    regime_features = build_regime_features(spy_daily)
    X, Y = build_factor_timing_dataset(factor_returns, regime_features)

    options_signals = _load_options_signals(cfg, spy_daily)
    options_signals = options_signals.copy()
    if not options_signals.empty:
        options_signals["month_end"] = pd.to_datetime(options_signals["month_end"])
        options_signals = options_signals.set_index("month_end").sort_index()

    months = sorted(X.index.unique())
    start = pd.to_datetime(cfg.backtest.test_start)
    months = [m for m in months if m >= start]

    records = []
    alloc_records: Dict[pd.Timestamp, pd.Series] = {}
    stock_weights_records: Dict[pd.Timestamp, pd.Series] = {}
    adjusted_stock_weights_records: Dict[pd.Timestamp, pd.Series] = {}
    scenario_weight_records: Dict[str, Dict[pd.Timestamp, pd.Series]] = {
        "unhedged": {},
        **{name: {} for name in _overlay_scenarios(cfg)},
        "score_scaled": {},
        "vol_benchmark": {},
        "hybrid_overlay": {},
    }

    predictions_list = []
    actuals_list = []

    prev_base_weights: Optional[pd.Series] = None
    prev_scenario_weights: Dict[str, Optional[pd.Series]] = {
        "unhedged": None,
        **{name: None for name in _overlay_scenarios(cfg)},
        "score_scaled": None,
        "vol_benchmark": None,
        "hybrid_overlay": None,
    }
    prev_warning_flag = 0
    base_returns_history: list[float] = []
    scenario_returns_history: Dict[str, list[float]] = {
        "unhedged": [],
        **{name: [] for name in _overlay_scenarios(cfg)},
        "score_scaled": [],
        "vol_benchmark": [],
        "hybrid_overlay": [],
    }
    overlay_scenarios = _overlay_scenarios(cfg)

    min_train_samples = 24

    for month in months:
        train_mask = X.index < month
        if train_mask.sum() < min_train_samples:
            continue

        X_train = X[train_mask]
        Y_train = Y[train_mask]
        model = train_factor_timing_model(cfg.factor_timing, X_train, Y_train)

        X_t = X.loc[[month]]
        pred_fac = predict_factor_returns(model, X_t, cfg.factor_timing.factors)
        preds_series = pd.Series(pred_fac.iloc[0].values, index=cfg.factor_timing.factors)

        if month in Y.index:
            predictions_list.append(preds_series.values)
            actuals_list.append(Y.loc[month].values)

        factor_alloc = compute_factor_allocations(cfg.factor_timing, preds_series)
        alloc_records[month] = factor_alloc

        target_weights_series = combine_factor_and_stock_weights(factor_alloc, factor_stock_weights, month)
        if target_weights_series.empty:
            continue

        if prev_base_weights is not None and cfg.factor_timing.allocation_smoothing < 1.0:
            gamma = cfg.factor_timing.allocation_smoothing
            aligned_prev = prev_base_weights.reindex(target_weights_series.index, fill_value=0.0)
            base_weights = gamma * target_weights_series + (1 - gamma) * aligned_prev
            if base_weights.abs().sum() > 0:
                base_weights = base_weights / base_weights.abs().sum()
        else:
            base_weights = target_weights_series

        if prev_base_weights is not None:
            aligned_prev = prev_base_weights.reindex(base_weights.index, fill_value=0.0)
            base_turnover = (base_weights - aligned_prev).abs().sum()
        else:
            base_turnover = base_weights.abs().sum()

        stock_weights_records[month] = base_weights

        panel_slice = panel_6[panel_6["month_end"] == month].set_index("ticker")
        fwd = panel_slice["forward_1m_return"]
        base_ret_gross = (base_weights.reindex(fwd.index).fillna(0.0) * fwd).sum()
        base_tc = (cfg.backtest.transaction_cost_bps / 10000.0) * base_turnover
        base_ret_net_pre_vol = base_ret_gross - base_tc
        base_vol_scalar = _compute_vol_scalar(base_returns_history, cfg)
        base_ret_net = base_ret_net_pre_vol * base_vol_scalar
        scenario_weight_records["unhedged"][month] = base_weights
        prev_unhedged_weights = prev_scenario_weights["unhedged"]

        warning_row = options_signals.loc[month] if month in options_signals.index else None
        warning_flag = int(warning_row["warning_flag"]) if warning_row is not None and pd.notna(warning_row["warning_flag"]) else 0
        warning_score = float(warning_row["warning_score"]) if warning_row is not None and pd.notna(warning_row["warning_score"]) else np.nan
        fixed_lambda_t = float(warning_row["lambda_t"]) if warning_row is not None and pd.notna(warning_row["lambda_t"]) else 1.0
        score_scaled_lambda_t = (
            float(warning_row["score_scaled_lambda_t"])
            if warning_row is not None and "score_scaled_lambda_t" in warning_row.index and pd.notna(warning_row["score_scaled_lambda_t"])
            else 1.0
        )
        vol_benchmark_lambda_t = _compute_vol_benchmark_lambda(base_returns_history, cfg)
        hybrid_overlay_lambda_t = min(vol_benchmark_lambda_t, vol_benchmark_lambda_t * score_scaled_lambda_t)
        record = {
            "month_end": month,
            "ret_gross": base_ret_gross,
            "ret_net": base_ret_net,
            "turnover": base_turnover,
            "warning_flag": warning_flag,
            "warning_score": warning_score,
            "base_vol_scalar": base_vol_scalar,
            "fixed_overlay_lambda_t": fixed_lambda_t,
            "score_scaled_lambda_t": score_scaled_lambda_t,
            "vol_benchmark_lambda_t": vol_benchmark_lambda_t,
            "hybrid_overlay_lambda_t": hybrid_overlay_lambda_t,
        }

        if prev_unhedged_weights is not None:
            aligned_prev_base = prev_unhedged_weights.reindex(base_weights.index, fill_value=0.0)
            unhedged_turnover = (base_weights - aligned_prev_base).abs().sum()
        else:
            unhedged_turnover = base_weights.abs().sum()
        record["unhedged_turnover"] = unhedged_turnover

        regime_switch_cost = 0.0
        if warning_flag != prev_warning_flag:
            regime_switch_cost = cfg.options_risk.state_change_cost_bps / 10000.0

        scenario_lambdas = {
            **{scenario_name: haircut if warning_flag == 1 else 1.0 for scenario_name, haircut in overlay_scenarios.items()},
            "score_scaled": score_scaled_lambda_t,
            "vol_benchmark": vol_benchmark_lambda_t,
            "hybrid_overlay": hybrid_overlay_lambda_t,
        }

        for scenario_name, scenario_lambda in scenario_lambdas.items():
            scenario_weights = base_weights * scenario_lambda
            scenario_weight_records[scenario_name][month] = scenario_weights

            prev_weights = prev_scenario_weights[scenario_name]
            if prev_weights is not None:
                aligned_prev = prev_weights.reindex(scenario_weights.index, fill_value=0.0)
                scenario_turnover = (scenario_weights - aligned_prev).abs().sum()
            else:
                scenario_turnover = scenario_weights.abs().sum()

            scenario_ret_gross = (scenario_weights.reindex(fwd.index).fillna(0.0) * fwd).sum()
            scenario_tc = (cfg.backtest.transaction_cost_bps / 10000.0) * scenario_turnover
            scenario_ret_net_pre_vol = scenario_ret_gross - scenario_tc - regime_switch_cost
            scenario_vol_scalar = _compute_vol_scalar(scenario_returns_history[scenario_name], cfg)
            scenario_ret_net = scenario_ret_net_pre_vol * scenario_vol_scalar

            record[f"{scenario_name}_lambda_t"] = scenario_lambda
            record[f"{scenario_name}_ret_gross"] = scenario_ret_gross
            record[f"{scenario_name}_ret_net"] = scenario_ret_net
            record[f"{scenario_name}_turnover"] = scenario_turnover
            record[f"{scenario_name}_regime_switch_cost"] = regime_switch_cost
            record[f"{scenario_name}_vol_scalar"] = scenario_vol_scalar

            scenario_returns_history[scenario_name].append(float(scenario_ret_net_pre_vol))

        adjusted_stock_weights_records[month] = scenario_weight_records["fixed_overlay"][month]
        record["ret_gross_overlay"] = record["fixed_overlay_ret_gross"]
        record["ret_net_overlay"] = record["fixed_overlay_ret_net"]
        record["overlay_turnover"] = record["fixed_overlay_turnover"]
        record["lambda_t"] = record["fixed_overlay_lambda_t"]
        record["regime_switch_cost"] = record["fixed_overlay_regime_switch_cost"]
        record["overlay_vol_scalar"] = record["fixed_overlay_vol_scalar"]

        records.append(record)

        prev_base_weights = base_weights.copy()
        prev_scenario_weights["unhedged"] = base_weights.copy()
        for scenario_name in scenario_lambdas:
            prev_scenario_weights[scenario_name] = scenario_weight_records[scenario_name][month].copy()
        prev_warning_flag = warning_flag
        base_returns_history.append(float(base_ret_net_pre_vol))

    if not records:
        raise RuntimeError("No backtest records produced; check data coverage.")

    monthly_returns = pd.DataFrame(records).set_index("month_end").sort_index()

    if alloc_records:
        alloc_df = pd.DataFrame(alloc_records).T.sort_index().reindex(columns=cfg.factor_timing.factors)
    else:
        alloc_df = pd.DataFrame(index=monthly_returns.index, columns=cfg.factor_timing.factors)

    if stock_weights_records:
        stock_weights_df = pd.DataFrame(stock_weights_records).T.sort_index()
        stock_weights_df = stock_weights_df.reindex(columns=cfg.demo_tickers, fill_value=0.0)
    else:
        stock_weights_df = pd.DataFrame(index=monthly_returns.index, columns=cfg.demo_tickers)

    if adjusted_stock_weights_records:
        adjusted_stock_weights_df = pd.DataFrame(adjusted_stock_weights_records).T.sort_index()
        adjusted_stock_weights_df = adjusted_stock_weights_df.reindex(columns=cfg.demo_tickers, fill_value=0.0)
    else:
        adjusted_stock_weights_df = pd.DataFrame(index=monthly_returns.index, columns=cfg.demo_tickers)

    scenario_stock_weights_df: Dict[str, pd.DataFrame] = {}
    for scenario_name, weight_records in scenario_weight_records.items():
        if weight_records:
            scenario_df = pd.DataFrame(weight_records).T.sort_index()
            scenario_df = scenario_df.reindex(columns=cfg.demo_tickers, fill_value=0.0)
        else:
            scenario_df = pd.DataFrame(index=monthly_returns.index, columns=cfg.demo_tickers)
        scenario_stock_weights_df[scenario_name] = scenario_df

    daily_returns = _build_daily_scenario_returns(cfg, monthly_returns, daily_prices, scenario_stock_weights_df)

    prediction_metrics = _compute_prediction_metrics(predictions_list, actuals_list)
    turnover_series = monthly_returns["turnover"]
    overlay_turnover_series = monthly_returns["overlay_turnover"]

    logger.info("Dynamic factor backtest completed with %d months", len(monthly_returns))
    logger.info("Average monthly turnover: %.2f%%", turnover_series.mean() * 100)
    logger.info("Average monthly overlay turnover: %.2f%%", overlay_turnover_series.mean() * 100)

    return BacktestResult(
        monthly_returns=monthly_returns,
        daily_returns=daily_returns,
        factor_allocations=alloc_df,
        stock_weights=stock_weights_df,
        adjusted_stock_weights=adjusted_stock_weights_df,
        scenario_stock_weights=scenario_stock_weights_df,
        prediction_metrics=prediction_metrics,
        turnover_series=turnover_series,
        overlay_turnover_series=overlay_turnover_series,
        options_signals=options_signals.reset_index() if not options_signals.empty else pd.DataFrame(),
    )



def _compute_prediction_metrics(predictions_list, actuals_list) -> Dict[str, float]:
    if not predictions_list or not actuals_list:
        return {"oos_r2": np.nan, "ic": np.nan}

    preds = np.array(predictions_list)
    actuals = np.array(actuals_list)

    preds_flat = preds.flatten()
    actuals_flat = actuals.flatten()

    mask = ~(np.isnan(preds_flat) | np.isnan(actuals_flat))
    preds_clean = preds_flat[mask]
    actuals_clean = actuals_flat[mask]

    if len(preds_clean) == 0:
        return {"oos_r2": np.nan, "ic": np.nan}

    ss_res = np.sum((actuals_clean - preds_clean) ** 2)
    ss_tot = np.sum((actuals_clean - np.mean(actuals_clean)) ** 2)
    oos_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    ic = np.corrcoef(preds_clean, actuals_clean)[0, 1]

    return {"oos_r2": float(oos_r2), "ic": float(ic)}
