from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from src.config import get_default_config  # noqa: E402
from src.data_download import download_ohlcv  # noqa: E402
from src.evaluation import (  # noqa: E402
    build_benchmark_return_frames,
    summarize_backtest,
    summarize_downturn_days,
    summarize_focus_hedge_ratios,
    summarize_hedge_tradeoff,
    summarize_overlay_scenarios,
    summarize_strategy_vs_benchmarks,
)
from src.features import build_monthly_feature_panel  # noqa: E402
from src.labels import add_forward_return_and_labels  # noqa: E402
from src.plots import (  # noqa: E402
    plot_demo_universe_panel,
    plot_drawdown_series,
    plot_drawdown_with_overlay,
    plot_daily_pnl_distributions,
    plot_focus_hedge_ratio_equity,
    plot_dynamic_portfolio_equity,
    plot_equity_with_overlay,
    plot_exposure_multipliers,
    plot_factor_weights_over_time,
    plot_gross_vs_net_equity,
    plot_left_tail_daily_distributions,
    plot_main_overlay_drawdowns,
    plot_main_overlay_equity_curves,
    plot_overlay_scenario_equity,
    plot_overlay_scenario_metrics,
    plot_rolling_sharpe,
    plot_stock_weights_over_time,
    plot_strategy_vs_benchmarks,
    plot_warning_score,
    plot_warning_score_vs_hedge_intensity,
)
from src.backtest import run_walk_forward_backtest  # noqa: E402
from src.model_comparison import run_model_comparison  # noqa: E402
from src.utils import ensure_directories, get_logger  # noqa: E402


def _write_csv_with_fallback(df, path: Path, logger, **kwargs) -> Path:
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        fallback_path = path.with_name(f"{path.stem}_latest{path.suffix}")
        logger.warning("Could not write %s because it is locked; writing %s instead", path, fallback_path)
        df.to_csv(fallback_path, **kwargs)
        return fallback_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run factor timing pipeline")
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Run model comparison mode (ridge, elasticnet, xgboost)",
    )
    args = parser.parse_args()

    cfg = get_default_config()
    logger = get_logger()

    ensure_directories(cfg.paths.data_raw, cfg.paths.data_processed, cfg.paths.outputs)

    tickers = cfg.demo_tickers
    logger.info("Using 6-stock demo universe: %s", ", ".join(tickers))

    prices, spy = download_ohlcv(cfg, tickers)

    feature_panel = build_monthly_feature_panel(cfg, prices, spy)
    labeled_panel = add_forward_return_and_labels(cfg, feature_panel)

    panel_path = cfg.paths.data_processed / "monthly_panel.parquet"
    labeled_panel.to_parquet(panel_path, index=False)
    logger.info("Saved monthly panel to %s", panel_path)

    if args.compare_models:
        logger.info("=" * 60)
        logger.info("RUNNING MODEL COMPARISON MODE")
        logger.info("=" * 60)
        run_model_comparison(cfg, labeled_panel, prices, spy, models=["ridge", "elasticnet", "xgboost"])
        plot_demo_universe_panel(cfg, labeled_panel, cfg.paths.outputs)
        return

    logger.info("=" * 60)
    logger.info("RUNNING DEFAULT MODEL: %s", cfg.factor_timing.model_name.upper())
    logger.info("=" * 60)

    bt_result = run_walk_forward_backtest(cfg, labeled_panel, prices, spy)
    bt_df = bt_result.monthly_returns.copy()
    summary_dir = cfg.paths.outputs

    summarize_backtest(
        bt_df,
        summary_dir,
        bt_result.prediction_metrics,
        bt_result.turnover_series,
    )
    monthly_overlay_summary, daily_overlay_summary = summarize_overlay_scenarios(
        bt_df,
        summary_dir,
        bt_result.daily_returns,
    )
    benchmark_monthly, benchmark_daily = build_benchmark_return_frames(
        labeled_panel,
        prices,
        spy,
        bt_df.index,
        bt_result.daily_returns.index if not bt_result.daily_returns.empty else None,
        cfg.demo_tickers,
    )
    benchmark_monthly_summary, benchmark_daily_summary = summarize_strategy_vs_benchmarks(
        bt_df,
        bt_result.daily_returns,
        benchmark_monthly,
        benchmark_daily,
        summary_dir,
    )
    summarize_focus_hedge_ratios(monthly_overlay_summary, daily_overlay_summary, summary_dir)
    downturn_summary = summarize_downturn_days(bt_result.daily_returns, spy, summary_dir)
    summarize_hedge_tradeoff(monthly_overlay_summary, daily_overlay_summary, downturn_summary, summary_dir)

    plot_dynamic_portfolio_equity(cfg, bt_df, summary_dir)
    plot_factor_weights_over_time(cfg, bt_result.factor_allocations, summary_dir)
    plot_stock_weights_over_time(bt_result.stock_weights, summary_dir)
    plot_gross_vs_net_equity(bt_df, summary_dir)
    plot_drawdown_series(bt_df, summary_dir)
    plot_rolling_sharpe(bt_df, summary_dir, window=12)
    plot_demo_universe_panel(cfg, labeled_panel, summary_dir)
    plot_warning_score(bt_result.options_signals, summary_dir)
    plot_exposure_multipliers(bt_df, summary_dir)
    plot_equity_with_overlay(bt_df, summary_dir)
    plot_drawdown_with_overlay(bt_df, summary_dir)
    plot_main_overlay_equity_curves(bt_df, summary_dir)
    plot_main_overlay_drawdowns(bt_df, summary_dir)
    plot_focus_hedge_ratio_equity(bt_df, summary_dir)
    plot_overlay_scenario_equity(bt_df, summary_dir)
    plot_overlay_scenario_metrics(monthly_overlay_summary, daily_overlay_summary, summary_dir)
    plot_strategy_vs_benchmarks(
        pd.read_csv(summary_dir / "strategy_vs_benchmark_monthly_returns.csv", index_col=0, parse_dates=True),
        summary_dir,
    )
    plot_daily_pnl_distributions(bt_result.daily_returns, summary_dir)
    plot_left_tail_daily_distributions(bt_result.daily_returns, summary_dir)
    plot_warning_score_vs_hedge_intensity(bt_result.options_signals, summary_dir)

    bt_result.factor_allocations.to_csv(summary_dir / "factor_allocations.csv", index=True)
    bt_result.stock_weights.to_csv(summary_dir / "stock_weights.csv", index=True)
    bt_result.adjusted_stock_weights.to_csv(summary_dir / "adjusted_stock_weights.csv", index=True)
    for scenario_name, scenario_weights in bt_result.scenario_stock_weights.items():
        scenario_weights.to_csv(summary_dir / f"{scenario_name}_stock_weights.csv", index=True)

    if not bt_result.options_signals.empty:
        _write_csv_with_fallback(bt_result.options_signals, summary_dir / "monthly_options_signals.csv", logger, index=False)
        _write_csv_with_fallback(
            bt_result.options_signals[[
                "month_end",
                "warning_score",
                "warning_threshold",
                "warning_flag",
                "fixed_hedge_budget",
                "fixed_overlay_equity_allocation",
                "score_mild_threshold",
                "score_high_threshold",
                "score_extreme_threshold",
                "score_scaled_hedge_budget",
                "score_scaled_equity_allocation",
                "market_stress_flag",
            ]],
            summary_dir / "warning_flags.csv",
            logger,
            index=False,
        )
        _write_csv_with_fallback(
            bt_result.options_signals[[
                "month_end",
                "warning_score",
                "warning_threshold",
                "warning_flag",
                "fixed_hedge_budget",
                "fixed_overlay_equity_allocation",
                "score_mild_threshold",
                "score_high_threshold",
                "score_extreme_threshold",
                "score_scaled_hedge_budget",
                "score_scaled_equity_allocation",
                "score_scaled_flag",
                "market_stress_flag",
                "realized_vol_3m",
                "drawdown",
            ]],
            summary_dir / "score_scaled_overlay_details.csv",
            logger,
            index=False,
        )
    if not bt_result.put_hedge_book.empty:
        _write_csv_with_fallback(
            bt_result.put_hedge_book,
            summary_dir / "monthly_put_hedge_book.csv",
            logger,
            index=False,
        )

    _write_csv_with_fallback(
        bt_df[[
            "ret_gross",
            "ret_net",
            "under_hedged_ret_net",
            "fixed_overlay_ret_net",
            "score_scaled_ret_net",
            "ret_gross_overlay",
            "ret_net_overlay",
            "over_hedged_ret_net",
            "turnover",
            "overlay_turnover",
            "warning_flag",
            "warning_score",
            "hedge_budget",
            "overlay_equity_allocation",
            "fixed_overlay_hedge_budget",
            "score_scaled_hedge_budget",
            "fixed_overlay_equity_allocation",
            "score_scaled_equity_allocation",
            "put_option_trade_available",
            "put_option_unit_total_return",
            "regime_switch_cost",
        ]],
        summary_dir / "monthly_returns_with_risk_overlay.csv",
        logger,
        index=True,
    )

    if not bt_result.daily_returns.empty:
        _write_csv_with_fallback(bt_result.daily_returns, summary_dir / "daily_pnl_returns.csv", logger, index=True)

    _write_csv_with_fallback(benchmark_monthly_summary, summary_dir / "strategy_vs_benchmark_monthly_summary.csv", logger, index=True)
    if not benchmark_daily_summary.empty:
        _write_csv_with_fallback(benchmark_daily_summary, summary_dir / "strategy_vs_benchmark_daily_summary.csv", logger, index=True)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
