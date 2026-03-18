from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .utils import get_logger, save_json


def _performance_stats(ret: pd.Series, periods_per_year: int = 12) -> Dict[str, float]:
    """
    Compute comprehensive performance statistics.
    
    Includes:
    - Annual return, volatility, Sharpe ratio
    - Sortino ratio (downside deviation)
    - Calmar ratio (CAGR / |MaxDD|)
    - CVaR (95% monthly)
    - Win rate
    - Skewness, kurtosis
    - Max drawdown
    """
    r = ret.dropna()
    if r.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "cvar_95": np.nan,
            "win_rate": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "max_drawdown": np.nan,
        }

    avg = r.mean()
    vol = r.std(ddof=0)
    compounded = (1.0 + r).prod()
    ann_return = compounded ** (periods_per_year / len(r)) - 1.0 if len(r) > 0 and compounded > 0 else np.nan
    ann_vol = vol * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    # Sortino ratio (downside deviation)
    downside = r[r < 0]
    downside_vol = downside.std(ddof=0) * np.sqrt(periods_per_year) if len(downside) > 0 else np.nan
    sortino = ann_return / downside_vol if downside_vol > 0 else np.nan

    # Max drawdown
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = dd.min()
    
    # Calmar ratio
    cagr = ann_return
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    # CVaR (95% monthly)
    cvar_95 = r.quantile(0.05) if len(r) > 0 else np.nan
    
    # Win rate
    win_rate = (r > 0).sum() / len(r) if len(r) > 0 else np.nan
    
    # Skewness and kurtosis
    from scipy.stats import skew, kurtosis as kurt
    skewness = skew(r.values, nan_policy='omit') if len(r) > 2 else np.nan
    kurtosis_val = kurt(r.values, nan_policy='omit') if len(r) > 3 else np.nan

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "cvar_95": float(cvar_95),
        "win_rate": float(win_rate),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis_val),
        "max_drawdown": float(max_dd),
    }


def summarize_backtest(
    monthly_returns: pd.DataFrame,
    outputs_dir,
    prediction_metrics: Dict[str, float] = None,
    turnover_series: pd.Series = None,
) -> Dict[str, float]:
    """
    Summarise dynamic factor portfolio performance.
    Returns the stats dict for use in model comparison.
    
    Computes stats for both gross and net returns.
    """
    logger = get_logger()
    monthly_returns = monthly_returns.copy()

    # Compute stats for net returns (primary metric)
    if "ret_net" in monthly_returns.columns:
        stats = _performance_stats(monthly_returns["ret_net"])
        stats_gross = _performance_stats(monthly_returns["ret_gross"])
        
        # Add gross return metrics with prefix
        for key, val in stats_gross.items():
            stats[f"gross_{key}"] = val
    else:
        # Backward compatibility: if only ret_dynamic_factor exists
        ret_col = "ret_dynamic_factor" if "ret_dynamic_factor" in monthly_returns.columns else monthly_returns.columns[0]
        stats = _performance_stats(monthly_returns[ret_col])
    
    # Add prediction metrics if provided
    if prediction_metrics:
        stats.update(prediction_metrics)
    
    # Add average turnover
    if turnover_series is not None:
        stats["avg_turnover"] = float(turnover_series.mean())

    summary = {
        "dynamic_factor_portfolio": stats,
    }

    outputs_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, outputs_dir / "backtest_summary.json")
    monthly_returns.to_csv(outputs_dir / "monthly_returns.csv", index=True)
    
    # Save turnover separately
    if turnover_series is not None:
        turnover_series.to_csv(outputs_dir / "turnover.csv", index=True, header=["turnover"])

    logger.info("Saved backtest summary and monthly returns to %s", outputs_dir)
    return stats


def build_overlay_scenario_frames(
    monthly_returns: pd.DataFrame,
    daily_returns: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_scenarios = pd.DataFrame(index=monthly_returns.index)
    monthly_scenarios["unhedged"] = monthly_returns["ret_net"]
    for column in monthly_returns.columns:
        if not column.endswith("_ret_net"):
            continue
        scenario_name = column.removesuffix("_ret_net")
        monthly_scenarios[scenario_name] = monthly_returns[column]

    daily_scenarios = daily_returns.copy() if daily_returns is not None else pd.DataFrame()
    return monthly_scenarios, daily_scenarios


def _scenario_monthly_metadata(monthly_returns: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    metadata = {
        "unhedged": {
            "lambda_col": None,
            "turnover_col": "turnover",
        }
    }
    for column in monthly_returns.columns:
        if not column.endswith("_ret_net"):
            continue
        scenario = column.removesuffix("_ret_net")
        metadata[scenario] = {
            "lambda_col": f"{scenario}_lambda_t" if f"{scenario}_lambda_t" in monthly_returns.columns else None,
            "turnover_col": f"{scenario}_turnover" if f"{scenario}_turnover" in monthly_returns.columns else None,
        }
    return metadata


def _operational_stats(
    scenario: str,
    monthly_returns: pd.DataFrame,
    metadata: Dict[str, str],
) -> Dict[str, float]:
    if scenario == "unhedged":
        lambdas = pd.Series(1.0, index=monthly_returns.index)
    elif metadata.get("lambda_col") and metadata["lambda_col"] in monthly_returns.columns:
        lambdas = monthly_returns[metadata["lambda_col"]].fillna(1.0)
    else:
        lambdas = pd.Series(np.nan, index=monthly_returns.index)

    if metadata.get("turnover_col") and metadata["turnover_col"] in monthly_returns.columns:
        turnover = monthly_returns[metadata["turnover_col"]].dropna()
    else:
        turnover = pd.Series(dtype=float)

    active_mask = lambdas < 0.999999
    active_lambdas = lambdas[active_mask]

    return {
        "ops_activation_count": float(active_mask.sum()),
        "ops_activation_rate": float(active_mask.mean()) if len(active_mask) > 0 else np.nan,
        "ops_avg_hedge_size_active": float((1.0 - active_lambdas).mean()) if not active_lambdas.empty else 0.0,
        "ops_avg_exposure_multiplier": float(lambdas.mean()) if not lambdas.empty else np.nan,
        "ops_avg_turnover": float(turnover.mean()) if not turnover.empty else np.nan,
    }


def summarize_overlay_scenarios(
    monthly_returns: pd.DataFrame,
    outputs_dir,
    daily_returns: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    monthly_scenarios, daily_scenarios = build_overlay_scenario_frames(monthly_returns, daily_returns)
    monthly_metadata = _scenario_monthly_metadata(monthly_returns)

    monthly_records = {}
    for name in monthly_scenarios.columns:
        perf = _performance_stats(monthly_scenarios[name], periods_per_year=12)
        monthly_records[name] = {
            "return_ann_return": perf["ann_return"],
            "return_ann_vol": perf["ann_vol"],
            "return_sharpe": perf["sharpe"],
            "return_sortino": perf["sortino"],
            "return_max_drawdown": perf["max_drawdown"],
            "return_cvar_95": perf["cvar_95"],
            "tail_skewness": perf["skewness"],
            "tail_kurtosis": perf["kurtosis"],
            **_operational_stats(name, monthly_returns, monthly_metadata.get(name, {})),
        }
    monthly_summary = pd.DataFrame(monthly_records).T
    monthly_summary.index.name = "scenario"
    monthly_summary.to_csv(outputs_dir / "overlay_scenario_comparison.csv")

    if daily_scenarios is not None and not daily_scenarios.empty:
        daily_records = {}
        for name in daily_scenarios.columns:
            perf = _performance_stats(daily_scenarios[name], periods_per_year=252)
            daily_records[name] = {
                "return_ann_return": perf["ann_return"],
                "return_ann_vol": perf["ann_vol"],
                "return_sharpe": perf["sharpe"],
                "return_sortino": perf["sortino"],
                "return_max_drawdown": perf["max_drawdown"],
                "return_cvar_95": perf["cvar_95"],
                "tail_skewness": perf["skewness"],
                "tail_kurtosis": perf["kurtosis"],
                **_operational_stats(name, monthly_returns, monthly_metadata.get(name, {})),
            }
        daily_summary = pd.DataFrame(daily_records).T
        daily_summary.index.name = "scenario"
        daily_scenarios.to_csv(outputs_dir / "daily_pnl_returns_by_scenario.csv", index=True)
        daily_summary.to_csv(outputs_dir / "daily_overlay_scenario_comparison.csv")
    else:
        daily_summary = pd.DataFrame()

    logger.info("Saved overlay scenario comparison tables to %s", outputs_dir)
    return monthly_summary, daily_summary


def summarize_focus_hedge_ratios(
    monthly_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    outputs_dir,
) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    focus_order = ["unhedged", "under_hedged", "fixed_overlay", "over_hedged", "score_scaled", "vol_benchmark"]

    combined = monthly_summary.add_prefix("monthly_").join(daily_summary.add_prefix("daily_"), how="left")
    focus = combined.loc[[name for name in focus_order if name in combined.index]].copy()
    focus.index.name = "scenario"
    focus.to_csv(outputs_dir / "hedge_ratio_focus_comparison.csv")
    return focus


def summarize_downturn_days(
    daily_returns: pd.DataFrame,
    spy_daily: pd.DataFrame,
    outputs_dir,
) -> pd.DataFrame:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if daily_returns is None or daily_returns.empty:
        return pd.DataFrame()

    spy = spy_daily.sort_values("date").copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy["spy_ret_d"] = spy["adj_close"].pct_change(fill_method=None)
    spy = spy.set_index("date")

    aligned = daily_returns.join(spy[["spy_ret_d"]], how="inner")
    if aligned.empty:
        return pd.DataFrame()

    worst_spy_cutoff = aligned["spy_ret_d"].quantile(0.10)
    masks = {
        "all_days": pd.Series(True, index=aligned.index),
        "spy_down_days": aligned["spy_ret_d"] < 0,
        "worst_10pct_spy_days": aligned["spy_ret_d"] <= worst_spy_cutoff,
    }

    records = []
    for scenario in daily_returns.columns:
        strategy_down_mask = aligned[scenario] < 0
        scenario_masks = {
            **masks,
            "strategy_down_days": strategy_down_mask,
        }
        for label, mask in scenario_masks.items():
            sample = aligned.loc[mask, scenario].dropna()
            if sample.empty:
                continue
            records.append(
                {
                    "scenario": scenario,
                    "subset": label,
                    "n_obs": int(sample.shape[0]),
                    "avg_return": float(sample.mean()),
                    "median_return": float(sample.median()),
                    "stdev": float(sample.std(ddof=0)),
                    "cvar_95": float(sample.quantile(0.05)),
                    "skewness": float(sample.skew()) if sample.shape[0] > 2 else np.nan,
                    "loss_rate": float((sample < 0).mean()),
                }
            )

    downturn_df = pd.DataFrame(records)
    if not downturn_df.empty:
        downturn_df.to_csv(outputs_dir / "downturn_day_comparison.csv", index=False)
        logger.info("Saved downturn-day comparison to %s", outputs_dir)
    return downturn_df


def summarize_hedge_tradeoff(
    monthly_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    downturn_df: pd.DataFrame,
    outputs_dir,
) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    focus = ["unhedged", "under_hedged", "fixed_overlay", "over_hedged", "score_scaled", "vol_benchmark"]
    rows = []
    for scenario in focus:
        if scenario not in monthly_summary.index:
            continue

        monthly_row = monthly_summary.loc[scenario]
        daily_row = daily_summary.loc[scenario] if scenario in daily_summary.index else pd.Series(dtype=float)
        downturn_slice = downturn_df[downturn_df["scenario"] == scenario] if downturn_df is not None and not downturn_df.empty else pd.DataFrame()

        def _subset_value(subset: str, column: str):
            if downturn_slice.empty:
                return np.nan
            matches = downturn_slice[downturn_slice["subset"] == subset]
            if matches.empty:
                return np.nan
            return float(matches.iloc[0][column])

        rows.append(
            {
                "scenario": scenario,
                "monthly_sharpe": float(monthly_row.get("return_sharpe", np.nan)),
                "monthly_ann_return": float(monthly_row.get("return_ann_return", np.nan)),
                "monthly_max_drawdown": float(monthly_row.get("return_max_drawdown", np.nan)),
                "daily_sharpe": float(daily_row.get("return_sharpe", np.nan)),
                "spy_down_day_avg_return": _subset_value("spy_down_days", "avg_return"),
                "worst_10pct_spy_day_avg_return": _subset_value("worst_10pct_spy_days", "avg_return"),
                "strategy_down_day_avg_return": _subset_value("strategy_down_days", "avg_return"),
                "daily_cvar_95": float(daily_row.get("return_cvar_95", np.nan)),
                "daily_skewness": float(daily_row.get("tail_skewness", np.nan)),
                "activation_count": float(monthly_row.get("ops_activation_count", np.nan)),
                "activation_rate": float(monthly_row.get("ops_activation_rate", np.nan)),
                "avg_hedge_size_active": float(monthly_row.get("ops_avg_hedge_size_active", np.nan)),
                "avg_exposure_multiplier": float(monthly_row.get("ops_avg_exposure_multiplier", np.nan)),
            }
        )

    summary = pd.DataFrame(rows).set_index("scenario")
    summary.to_csv(outputs_dir / "hedge_tradeoff_summary.csv")
    return summary

