from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import PipelineConfig
from .utils import get_logger


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def plot_dynamic_portfolio_equity(
    cfg: PipelineConfig,
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """
    Plot cumulative equity curve for the dynamic factor portfolio.
    Uses net returns if available, otherwise falls back to other columns.
    """
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Determine which return column to use
    if "ret_net" in monthly_returns.columns:
        ret_col = "ret_net"
        label = "Dynamic Factor Portfolio (Net)"
    elif "ret_dynamic_factor" in monthly_returns.columns:
        ret_col = "ret_dynamic_factor"
        label = "Dynamic Factor Portfolio"
    else:
        ret_col = monthly_returns.columns[0]
        label = "Dynamic Factor Portfolio"

    fig, ax = plt.subplots(figsize=(10, 5))
    eq = _equity_curve(monthly_returns[ret_col])
    ax.plot(eq.index, eq.values, label=label)
    ax.set_title("Cumulative Dynamic Factor Portfolio")
    ax.set_ylabel("Cumulative Growth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs_dir / "cumulative_dynamic_factor_portfolio.png", dpi=150)
    plt.close(fig)
    logger.info("Saved dynamic factor portfolio equity plot")


def plot_factor_weights_over_time(
    cfg: PipelineConfig,
    factor_allocations: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """
    Plot factor allocations over time as a heatmap.
    """
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if factor_allocations.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        factor_allocations.T,
        cmap="coolwarm",
        center=0.0,
        cbar_kws={"label": "Factor Weight"},
        ax=ax,
    )
    ax.set_title("Factor Weights Over Time")
    ax.set_ylabel("Factor")
    ax.set_xlabel("Month")
    fig.tight_layout()
    fig.savefig(outputs_dir / "factor_weights_over_time.png", dpi=150)
    plt.close(fig)
    logger.info("Saved factor weights over time plot")


def plot_demo_universe_panel(
    cfg: PipelineConfig,
    panel: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """
    Keep the demo universe cumulative forward return plot for teaching.
    """
    logger = get_logger()
    demo = panel[panel["ticker"].isin(cfg.demo_tickers)].copy()
    if demo.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for ticker, grp in demo.groupby("ticker"):
        eq = (1.0 + grp.set_index("month_end")["forward_1m_return"].fillna(0.0)).cumprod()
        ax.plot(eq.index, eq.values, label=ticker)
    ax.set_title("Demo Universe: Forward 1M Returns Cumulative")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs_dir / "demo_universe_cumulative_forward_returns.png", dpi=150)
    plt.close(fig)
    logger.info("Saved demo universe plot")




def plot_stock_weights_over_time(
    stock_weights: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """
    Plot stock weights over time as a heatmap (Part 4.1).
    """
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if stock_weights.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        stock_weights.T,
        cmap="RdYlGn",
        center=0.0,
        cbar_kws={"label": "Stock Weight"},
        ax=ax,
    )
    ax.set_title("Stock Weights Over Time")
    ax.set_ylabel("Ticker")
    ax.set_xlabel("Month")
    fig.tight_layout()
    fig.savefig(outputs_dir / "stock_weights_over_time.png", dpi=150)
    plt.close(fig)
    logger.info("Saved stock weights over time plot")


def plot_gross_vs_net_equity(
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """
    Plot gross vs net equity curves (Part 4.2).
    """
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if "ret_gross" not in monthly_returns.columns or "ret_net" not in monthly_returns.columns:
        logger.warning("Gross/net returns not found, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    
    eq_gross = _equity_curve(monthly_returns["ret_gross"])
    eq_net = _equity_curve(monthly_returns["ret_net"])
    
    ax.plot(eq_gross.index, eq_gross.values, label="Gross Returns", linewidth=2, alpha=0.8)
    ax.plot(eq_net.index, eq_net.values, label="Net Returns (after TC)", linewidth=2, alpha=0.8)
    
    ax.set_title("Gross vs Net Equity Curves")
    ax.set_ylabel("Cumulative Growth")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "equity_curve_gross_vs_net.png", dpi=150)
    plt.close(fig)
    logger.info("Saved gross vs net equity curve plot")


def plot_drawdown_series(
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """
    Plot drawdown time series (Part 4.3).
    """
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ret_col = "ret_net" if "ret_net" in monthly_returns.columns else monthly_returns.columns[0]
    ret = monthly_returns[ret_col]
    
    eq = _equity_curve(ret)
    peak = eq.cummax()
    dd = eq / peak - 1.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color="red")
    ax.plot(dd.index, dd.values, color="darkred", linewidth=1.5)
    ax.set_title("Drawdown Series")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "drawdown_series.png", dpi=150)
    plt.close(fig)
    logger.info("Saved drawdown series plot")


def plot_rolling_sharpe(
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
    window: int = 12,
) -> None:
    """
    Plot rolling Sharpe ratio (Part 4.4).
    """
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ret_col = "ret_net" if "ret_net" in monthly_returns.columns else monthly_returns.columns[0]
    ret = monthly_returns[ret_col]
    
    rolling_mean = ret.rolling(window=window).mean() * 12
    rolling_std = ret.rolling(window=window).std() * np.sqrt(12)
    rolling_sharpe = rolling_mean / rolling_std

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color="steelblue")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(y=1, color="green", linestyle=":", linewidth=0.8, alpha=0.5, label="Sharpe=1")
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / f"rolling_sharpe_{window}m.png", dpi=150)
    plt.close(fig)
    logger.info("Saved rolling Sharpe plot")


def plot_factor_return_contributions(
    monthly_returns: pd.DataFrame,
    factor_allocations: pd.DataFrame,
    factor_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """
    Plot factor return contributions over time (Part 4.5).
    
    contribution_{k,t} = alpha_{k,t} * r^{factor_k}_t
    """
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if factor_allocations.empty or factor_returns.empty:
        logger.warning("Factor allocations or returns empty, skipping contribution plot")
        return

    # Align indices
    common_idx = factor_allocations.index.intersection(factor_returns.index)
    if len(common_idx) == 0:
        logger.warning("No common dates for factor contributions, skipping plot")
        return
    
    alloc = factor_allocations.loc[common_idx]
    ret = factor_returns.loc[common_idx]
    
    # Compute contributions
    contributions = alloc * ret
    
    fig, ax = plt.subplots(figsize=(12, 6))
    contributions.plot.area(ax=ax, alpha=0.7, stacked=True)
    ax.set_title("Factor Return Contributions Over Time")
    ax.set_ylabel("Contribution to Portfolio Return")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(outputs_dir / "factor_return_contributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved factor return contributions plot")


def plot_model_comparison_equity_curves(
    outputs_dir: Path,
    models: List[str] = None,
) -> None:
    """
    Plot equity curves for all models on the same chart (Part 4.6).
    """
    logger = get_logger()
    
    if models is None:
        models = ["ridge", "elasticnet", "xgboost"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {"ridge": "#1f77b4", "elasticnet": "#ff7f0e", "xgboost": "#2ca02c"}
    
    for model_name in models:
        returns_path = outputs_dir / "model_comparison" / model_name / "monthly_returns.csv"
        
        if not returns_path.exists():
            logger.warning("Returns file not found for %s", model_name)
            continue
        
        returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        # Use net returns if available, otherwise fall back
        if "ret_net" in returns_df.columns:
            ret_col = "ret_net"
        elif "ret_dynamic_factor" in returns_df.columns:
            ret_col = "ret_dynamic_factor"
        else:
            ret_col = returns_df.columns[0]
        
        cum_returns = _equity_curve(returns_df[ret_col])
        
        ax.plot(
            cum_returns.index,
            cum_returns.values,
            label=model_name.capitalize(),
            linewidth=2,
            color=colors.get(model_name, None)
        )
    
    ax.set_title("Model Comparison: Equity Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = outputs_dir / "model_comparison" / "model_comparison_equity_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info("Saved model comparison equity curves to %s", output_path)


def generate_all_plots(
    cfg: PipelineConfig,
    bt_result,
    factor_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """
    Generate all plots for a single backtest run.
    """
    logger = get_logger()
    logger.info("Generating all plots...")
    
    # Original plots
    plot_dynamic_portfolio_equity(cfg, bt_result.monthly_returns, outputs_dir)
    plot_factor_weights_over_time(cfg, bt_result.factor_allocations, outputs_dir)
    
    # New plots (Part 4)
    plot_stock_weights_over_time(bt_result.stock_weights, outputs_dir)
    plot_gross_vs_net_equity(bt_result.monthly_returns, outputs_dir)
    plot_drawdown_series(bt_result.monthly_returns, outputs_dir)
    plot_rolling_sharpe(bt_result.monthly_returns, outputs_dir, window=12)
    plot_factor_return_contributions(
        bt_result.monthly_returns,
        bt_result.factor_allocations,
        factor_returns,
        outputs_dir
    )
    
    logger.info("All plots generated successfully")


def _drawdown_curve(returns: pd.Series) -> pd.Series:
    eq = _equity_curve(returns)
    peak = eq.cummax()
    return eq / peak - 1.0


def _overlay_monthly_scenarios(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    scenario_df = pd.DataFrame(index=monthly_returns.index)
    scenario_df["Unhedged"] = monthly_returns["ret_net"]
    preferred_order = ["under_hedged", "fixed_overlay", "over_hedged", "score_scaled", "vol_benchmark"]

    for scenario_name in preferred_order:
        column = f"{scenario_name}_ret_net"
        if column in monthly_returns.columns:
            scenario_df[scenario_name.replace("_", " ").title()] = monthly_returns[column]

    for column in sorted(monthly_returns.columns):
        if not column.endswith("_ret_net"):
            continue
        scenario_name = column.removesuffix("_ret_net")
        if scenario_name in preferred_order:
            continue
        scenario_df[scenario_name.replace("_", " ").upper()] = monthly_returns[column]

    return scenario_df


def _main_overlay_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    scenario_df = pd.DataFrame(index=monthly_returns.index)
    scenario_df["Unhedged"] = monthly_returns["ret_net"]
    mapping = {
        "Fixed overlay": "fixed_overlay_ret_net",
        "Score-scaled overlay": "score_scaled_ret_net",
        "Vol benchmark": "vol_benchmark_ret_net",
        "Hybrid overlay": "hybrid_overlay_ret_net",
    }
    for label, column in mapping.items():
        if column in monthly_returns.columns:
            scenario_df[label] = monthly_returns[column]
    return scenario_df


def _focus_hedge_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    scenario_df = pd.DataFrame(index=monthly_returns.index)
    mapping = {
        "Unhedged": "ret_net",
        "Under-hedged": "under_hedged_ret_net",
        "Well-hedged": "fixed_overlay_ret_net",
        "Over-hedged": "over_hedged_ret_net",
    }
    for label, column in mapping.items():
        if column in monthly_returns.columns:
            scenario_df[label] = monthly_returns[column]
    return scenario_df



def _shade_warning_regimes(ax, monthly_returns: pd.DataFrame) -> None:
    if "warning_flag" not in monthly_returns.columns:
        return
    warnings = monthly_returns["warning_flag"].fillna(0).astype(int)
    for date, flag in warnings.items():
        if flag == 1:
            ax.axvspan(date - pd.offsets.MonthBegin(1), date + pd.offsets.MonthEnd(0), color="gold", alpha=0.15)



def plot_warning_score(
    options_signals: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if options_signals.empty or "warning_score" not in options_signals.columns:
        logger.warning("Warning score data unavailable, skipping warning-score plot")
        return

    df = options_signals.copy()
    df["month_end"] = pd.to_datetime(df["month_end"])
    df = df.sort_values("month_end")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["month_end"], df["warning_score"], linewidth=2, color="darkorange", label="Warning score")
    if "warning_threshold" in df.columns:
        ax.plot(df["month_end"], df["warning_threshold"], linestyle="--", color="black", alpha=0.7, label="Threshold")
    ax.axhline(y=0, color="grey", linewidth=0.8, alpha=0.5)
    ax.set_title("Options-Based Warning Score")
    ax.set_ylabel("Score")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "warning_score_over_time.png", dpi=150)
    plt.close(fig)
    logger.info("Saved warning score plot")



def plot_equity_with_overlay(
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if "ret_net_overlay" not in monthly_returns.columns or "ret_net" not in monthly_returns.columns:
        logger.warning("Overlay return series unavailable, skipping overlay equity plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    base_eq = _equity_curve(monthly_returns["ret_net"])
    overlay_eq = _equity_curve(monthly_returns["ret_net_overlay"])

    ax.plot(base_eq.index, base_eq.values, linewidth=2, label="Equity strategy")
    ax.plot(overlay_eq.index, overlay_eq.values, linewidth=2, label="With options risk overlay")
    _shade_warning_regimes(ax, monthly_returns)
    ax.set_title("Equity Curve With and Without Options Risk Overlay")
    ax.set_ylabel("Cumulative Growth")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "equity_curve_with_vs_without_overlay.png", dpi=150)
    plt.close(fig)
    logger.info("Saved overlay equity comparison plot")



def plot_drawdown_with_overlay(
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if "ret_net_overlay" not in monthly_returns.columns or "ret_net" not in monthly_returns.columns:
        logger.warning("Overlay return series unavailable, skipping overlay drawdown plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    base_dd = _drawdown_curve(monthly_returns["ret_net"])
    overlay_dd = _drawdown_curve(monthly_returns["ret_net_overlay"])

    ax.plot(base_dd.index, base_dd.values, linewidth=2, label="Equity strategy", color="firebrick")
    ax.plot(overlay_dd.index, overlay_dd.values, linewidth=2, label="With options risk overlay", color="navy")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Drawdown With and Without Options Risk Overlay")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "drawdown_with_vs_without_overlay.png", dpi=150)
    plt.close(fig)
    logger.info("Saved overlay drawdown comparison plot")


def plot_overlay_scenario_equity(
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    scenario_returns = _overlay_monthly_scenarios(monthly_returns)
    if scenario_returns.empty:
        logger.warning("Overlay scenario returns unavailable, skipping scenario equity plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, series in scenario_returns.items():
        ax.plot(series.index, _equity_curve(series).values, linewidth=2, label=label)
    _shade_warning_regimes(ax, monthly_returns)
    ax.set_title("Overlay Sizing Comparison")
    ax.set_ylabel("Cumulative Growth")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "overlay_scenario_equity_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved overlay scenario equity comparison plot")


def plot_overlay_scenario_metrics(
    monthly_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if monthly_summary.empty:
        logger.warning("Scenario summary unavailable, skipping scenario metrics plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    chart_specs = [
        ("return_sharpe", monthly_summary, "Monthly Sharpe"),
        ("return_max_drawdown", monthly_summary, "Monthly Max Drawdown"),
        ("tail_skewness", daily_summary if not daily_summary.empty else monthly_summary, "Skewness"),
        ("return_cvar_95", daily_summary if not daily_summary.empty else monthly_summary, "CVaR 95%"),
    ]

    for ax, (metric, df, title) in zip(axes.flat, chart_specs):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        values = df[metric].sort_values()
        values.plot(kind="barh", ax=ax, color="#2f6c8f")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(outputs_dir / "overlay_scenario_metric_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved overlay scenario metric comparison plot")


def plot_daily_pnl_distributions(
    daily_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if daily_returns.empty:
        logger.warning("Daily returns unavailable, skipping daily distribution plot")
        return

    preferred = ["unhedged", "fixed_overlay", "score_scaled", "vol_benchmark"]
    selected = [col for col in preferred if col in daily_returns.columns]
    plot_df = daily_returns[selected] if selected else daily_returns

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)

    for column in plot_df.columns:
        sns.kdeplot(plot_df[column].dropna(), ax=axes[0], linewidth=2, label=column.replace("_", " ").title())
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Daily PnL Return Distributions")
    axes[0].set_xlabel("Daily Return")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    skewness = plot_df.skew().sort_values()
    skewness.plot(kind="barh", ax=axes[1], color="#b56576")
    axes[1].set_title("Daily Return Skewness by Scenario")
    axes[1].set_xlabel("Skewness")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outputs_dir / "daily_pnl_return_distribution_and_skew.png", dpi=150)
    plt.close(fig)
    logger.info("Saved daily return distribution and skew plot")


def plot_exposure_multipliers(
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    columns = {
        "Fixed overlay": "fixed_overlay_lambda_t",
        "Score-scaled overlay": "score_scaled_lambda_t",
        "Vol benchmark": "vol_benchmark_lambda_t",
        "Hybrid overlay": "hybrid_overlay_lambda_t",
    }
    available = {label: monthly_returns[col] for label, col in columns.items() if col in monthly_returns.columns}
    if not available:
        logger.warning("Exposure multiplier columns unavailable, skipping multiplier plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, series in available.items():
        ax.plot(series.index, series.values, linewidth=2, label=label)
    ax.set_title("Equity Capital Multiplier After Funding Put Hedge")
    ax.set_ylabel("Equity Allocation")
    ax.set_xlabel("Date")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "exposure_multiplier_over_time.png", dpi=150)
    plt.close(fig)
    logger.info("Saved exposure multiplier plot")


def plot_main_overlay_equity_curves(
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    scenario_returns = _main_overlay_returns(monthly_returns)
    if scenario_returns.empty:
        logger.warning("Main overlay returns unavailable, skipping main equity plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, series in scenario_returns.items():
        ax.plot(series.index, _equity_curve(series).values, linewidth=2, label=label)
    _shade_warning_regimes(ax, monthly_returns)
    ax.set_title("Main Overlay Comparison")
    ax.set_ylabel("Cumulative Growth")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "main_overlay_equity_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved main overlay equity plot")


def plot_main_overlay_drawdowns(
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    scenario_returns = _main_overlay_returns(monthly_returns)
    if scenario_returns.empty:
        logger.warning("Main overlay returns unavailable, skipping main drawdown plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, series in scenario_returns.items():
        ax.plot(series.index, _drawdown_curve(series).values, linewidth=2, label=label)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Main Overlay Drawdown Comparison")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "main_overlay_drawdown_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved main overlay drawdown comparison plot")


def plot_warning_score_vs_hedge_intensity(
    options_signals: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if options_signals.empty or "warning_score" not in options_signals.columns:
        logger.warning("Score-scaled overlay details unavailable, skipping score-vs-lambda plot")
        return

    df = options_signals.copy()
    if "score_scaled_hedge_budget" in df.columns:
        y = df["score_scaled_hedge_budget"]
        ylabel = "Score-scaled Hedge Budget"
    elif "score_scaled_lambda_t" in df.columns:
        y = df["score_scaled_lambda_t"]
        ylabel = "Score-scaled Lambda"
    else:
        logger.warning("No hedge-intensity column found, skipping score-vs-lambda plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["warning_score"], y, alpha=0.7, color="#2f6c8f")
    ax.set_title("Warning Score vs Applied Hedge Intensity")
    ax.set_xlabel("Warning Score")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "warning_score_vs_hedge_intensity.png", dpi=150)
    plt.close(fig)
    logger.info("Saved warning score vs hedge intensity plot")


def plot_focus_hedge_ratio_equity(
    monthly_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    scenario_returns = _focus_hedge_returns(monthly_returns)
    if scenario_returns.empty:
        logger.warning("Focused hedge ratio returns unavailable, skipping focus equity plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, series in scenario_returns.items():
        ax.plot(series.index, _equity_curve(series).values, linewidth=2, label=label)
    _shade_warning_regimes(ax, monthly_returns)
    ax.set_title("Under / Well / Over Hedged Comparison")
    ax.set_ylabel("Cumulative Growth")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_dir / "hedge_ratio_focus_equity.png", dpi=150)
    plt.close(fig)
    logger.info("Saved focused hedge-ratio equity plot")


def plot_left_tail_daily_distributions(
    daily_returns: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    logger = get_logger()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if daily_returns.empty:
        logger.warning("Daily returns unavailable, skipping left-tail daily distribution plot")
        return

    focus_cols = [col for col in ["unhedged", "under_hedged", "fixed_overlay", "over_hedged"] if col in daily_returns.columns]
    plot_df = daily_returns[focus_cols] if focus_cols else daily_returns
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for column in plot_df.columns:
        sns.histplot(plot_df[column].dropna(), bins=60, stat="density", element="step", fill=False, ax=axes[0], label=column.replace("_", " ").title())
    axes[0].set_title("Daily PnL Distribution")
    axes[0].set_xlabel("Daily Return")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    left_tail = plot_df[plot_df.le(0.0)].stack().reset_index(drop=True)
    tail_floor = float(left_tail.quantile(0.01)) if not left_tail.empty else float(plot_df.min().min())
    tail_ceiling = 0.0
    for column in plot_df.columns:
        sns.kdeplot(plot_df[column].dropna(), ax=axes[1], linewidth=2, label=column.replace("_", " ").title())
    axes[1].set_xlim(tail_floor, tail_ceiling)
    axes[1].set_title("Left-Tail Zoom of Daily PnL")
    axes[1].set_xlabel("Daily Return")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outputs_dir / "daily_pnl_left_tail_focus.png", dpi=150)
    plt.close(fig)
    logger.info("Saved left-tail daily PnL plot")
