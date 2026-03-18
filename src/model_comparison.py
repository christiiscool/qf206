from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .backtest import run_walk_forward_backtest
from .comparison_plots import create_all_comparison_plots
from .config import PipelineConfig
from .evaluation import summarize_backtest
from .plots import (
    plot_dynamic_portfolio_equity,
    plot_factor_weights_over_time,
    plot_stock_weights_over_time,
    plot_gross_vs_net_equity,
    plot_drawdown_series,
    plot_rolling_sharpe,
    plot_model_comparison_equity_curves,
)
from .utils import get_logger


def run_model_comparison(
    cfg: PipelineConfig,
    panel: pd.DataFrame,
    daily_prices: pd.DataFrame,
    spy_daily: pd.DataFrame,
    models: List[str] = None,
) -> pd.DataFrame:
    """
    Run backtests for multiple factor timing models and compare results.
    
    Args:
        cfg: Pipeline configuration
        panel: Monthly feature panel
        daily_prices: Daily asset prices
        spy_daily: Daily SPY data
        models: List of model names to test (default: ["ridge", "elasticnet", "xgboost"])
    
    Returns:
        DataFrame with comparison metrics for all models
    """
    logger = get_logger()
    
    if models is None:
        models = ["ridge", "elasticnet", "xgboost"]
    
    logger.info("Starting model comparison for models: %s", ", ".join(models))
    
    comparison_results = []
    
    for model_name in models:
        logger.info("=" * 60)
        logger.info("Running backtest for model: %s", model_name)
        logger.info("=" * 60)
        
        # Update config for this model
        cfg.factor_timing.model_name = model_name
        
        # Create model-specific output directory
        model_output_dir = cfg.paths.outputs / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run backtest
        bt_result = run_walk_forward_backtest(cfg, panel, daily_prices, spy_daily)
        
        # Save results
        stats = summarize_backtest(
            bt_result.monthly_returns,
            model_output_dir,
            bt_result.prediction_metrics,
            bt_result.turnover_series
        )
        
        # Save factor allocations and stock weights
        bt_result.factor_allocations.to_csv(
            model_output_dir / "factor_allocations.csv",
            index=True
        )
        bt_result.stock_weights.to_csv(
            model_output_dir / "stock_weights.csv",
            index=True
        )
        
        # Generate plots
        plot_dynamic_portfolio_equity(
            cfg,
            bt_result.monthly_returns,
            model_output_dir
        )
        plot_factor_weights_over_time(
            cfg,
            bt_result.factor_allocations,
            model_output_dir
        )
        plot_stock_weights_over_time(
            bt_result.stock_weights,
            model_output_dir
        )
        plot_gross_vs_net_equity(
            bt_result.monthly_returns,
            model_output_dir
        )
        plot_drawdown_series(
            bt_result.monthly_returns,
            model_output_dir
        )
        plot_rolling_sharpe(
            bt_result.monthly_returns,
            model_output_dir,
            window=12
        )
        
        # Collect results for comparison
        result_row = {
            "model": model_name,
            "annual_return": stats.get("ann_return", float("nan")),
            "annual_volatility": stats.get("ann_vol", float("nan")),
            "sharpe_ratio": stats.get("sharpe", float("nan")),
            "sortino_ratio": stats.get("sortino", float("nan")),
            "calmar_ratio": stats.get("calmar", float("nan")),
            "max_drawdown": stats.get("max_drawdown", float("nan")),
            "cvar_95": stats.get("cvar_95", float("nan")),
            "win_rate": stats.get("win_rate", float("nan")),
            "avg_turnover": stats.get("avg_turnover", float("nan")),
            "oos_r2": stats.get("oos_r2", float("nan")),
            "ic": stats.get("ic", float("nan")),
        }
        comparison_results.append(result_row)
        
        logger.info("Completed backtest for %s", model_name)
        logger.info("Annual Return: %.2f%%", stats.get("ann_return", 0) * 100)
        logger.info("Sharpe Ratio: %.2f", stats.get("sharpe", 0))
        logger.info("OOS R²: %.4f", stats.get("oos_r2", 0))
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.set_index("model")
    
    # Save comparison table
    comparison_path = cfg.paths.outputs / "model_comparison.csv"
    comparison_df.to_csv(comparison_path)
    
    # Generate comparison plots
    logger.info("Generating comparison visualizations...")
    create_all_comparison_plots(cfg.paths.outputs, comparison_df)
    plot_model_comparison_equity_curves(cfg.paths.outputs, models=models)
    
    logger.info("=" * 60)
    logger.info("Model Comparison Summary")
    logger.info("=" * 60)
    logger.info("\n%s", comparison_df.to_string())
    logger.info("\nComparison table saved to: %s", comparison_path)
    
    return comparison_df
