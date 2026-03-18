"""
Additional plotting functions for model comparison.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import get_logger


def plot_model_comparison_metrics(
    comparison_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Create visualization comparing model performance metrics.
    """
    logger = get_logger()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Model Comparison: Factor Timing Performance", fontsize=16, y=1.00)
    
    metrics = [
        ("annual_return", "Annual Return", "%.1f%%"),
        ("annual_volatility", "Annual Volatility", "%.1f%%"),
        ("sharpe_ratio", "Sharpe Ratio", "%.2f"),
        ("max_drawdown", "Max Drawdown", "%.1f%%"),
        ("oos_r2", "Out-of-Sample R²", "%.3f"),
        ("ic", "Information Coefficient", "%.3f"),
    ]
    
    for idx, (metric, title, fmt) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric not in comparison_df.columns:
            ax.text(0.5, 0.5, f"{metric}\nNot Available", 
                   ha="center", va="center", fontsize=12)
            ax.set_title(title)
            continue
        
        values = comparison_df[metric].copy()
        
        # Convert to percentage for certain metrics
        if metric in ["annual_return", "annual_volatility", "max_drawdown"]:
            values = values * 100
        
        # Create bar plot
        bars = ax.bar(range(len(values)), values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values.index, rotation=0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            label_y = height if height > 0 else height - abs(height) * 0.1
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   fmt % val,
                   ha="center", va="bottom" if height > 0 else "top",
                   fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    output_path = output_dir / "model_comparison_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info("Saved model comparison plot to %s", output_path)


def plot_cumulative_returns_comparison(
    output_dir: Path,
    models: List[str] = None,
) -> None:
    """
    Plot cumulative returns for all models on the same chart.
    """
    logger = get_logger()
    
    if models is None:
        models = ["ridge", "elasticnet", "xgboost"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {"ridge": "#1f77b4", "elasticnet": "#ff7f0e", "xgboost": "#2ca02c"}
    
    for model_name in models:
        returns_path = output_dir / model_name / "monthly_returns.csv"
        
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
            logger.warning("No returns column found for %s", model_name)
            continue
        
        cum_returns = (1 + returns_df[ret_col]).cumprod()
        
        ax.plot(
            cum_returns.index,
            cum_returns.values,
            label=model_name.capitalize(),
            linewidth=2,
            color=colors.get(model_name, None)
        )
    
    ax.set_title("Cumulative Returns Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "cumulative_returns_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info("Saved cumulative returns comparison to %s", output_path)


def create_all_comparison_plots(output_dir: Path, comparison_df: pd.DataFrame) -> None:
    """
    Generate all comparison visualizations.
    """
    plot_model_comparison_metrics(comparison_df, output_dir)
    plot_cumulative_returns_comparison(output_dir)
