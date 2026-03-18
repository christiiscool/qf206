from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import FactorTimingConfig, PipelineConfig
from .utils import get_logger


def compute_factor_allocations(
    cfg: FactorTimingConfig, predicted_factor_returns: pd.Series
) -> pd.Series:
    """
    Compute factor allocations from predicted factor returns.

    If allow_short_factors:
        alpha_k = pred_k / sum_j |pred_j|
        then clip to +/- max_factor_weight and renormalize.
    Else:
        alpha_k = max(pred_k, 0) normalized to sum 1, with cap at max_factor_weight.
    """
    logger = get_logger()
    preds = predicted_factor_returns.astype(float).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    if cfg.allow_short_factors:
        denom = np.abs(preds).sum()
        if denom <= 0:
            # fall back to equal-weight across factors
            alloc = pd.Series(1.0 / len(preds), index=preds.index)
        else:
            alloc = preds / denom
    else:
        non_neg = preds.clip(lower=0.0)
        total = non_neg.sum()
        if total <= 0:
            alloc = pd.Series(1.0 / len(preds), index=preds.index)
        else:
            alloc = non_neg / total

    # Apply max_factor_weight cap and renormalize
    max_w = cfg.max_factor_weight
    alloc = alloc.clip(lower=-max_w, upper=max_w)
    if cfg.allow_short_factors:
        total_abs = np.abs(alloc).sum()
        if total_abs > 0:
            alloc = alloc / total_abs
    else:
        total_pos = alloc.clip(lower=0.0).sum()
        if total_pos > 0:
            alloc = alloc / total_pos

    logger.debug("Factor allocations: %s", alloc.to_dict())
    return alloc


def combine_factor_and_stock_weights(
    factor_allocations: pd.Series,
    factor_stock_weights: Dict[str, pd.DataFrame],
    month_end,
) -> pd.Series:
    """
    Combine factor-level allocations with factor stock weights for a given month.

    factor_allocations: Series indexed by factor name.
    factor_stock_weights: dict[factor_name] -> DataFrame[month_end, ticker, weight].
    month_end: the month to construct weights for.
    """
    logger = get_logger()
    contribs = []

    for factor_name, alpha_k in factor_allocations.items():
        weights_df = factor_stock_weights.get(factor_name)
        if weights_df is None or weights_df.empty or alpha_k == 0.0:
            continue
        sl = weights_df[weights_df["month_end"] == month_end]
        if sl.empty:
            continue
        s = sl.set_index("ticker")["weight"] * alpha_k
        contribs.append(s)

    if not contribs:
        return pd.Series(dtype=float)

    total = contribs[0]
    for s in contribs[1:]:
        total = total.add(s, fill_value=0.0)

    # Normalize to sum to 1 (allowing negative weights if factors are short)
    if total.abs().sum() > 0:
        total = total / total.abs().sum()
    logger.debug("Final stock weights for %s: %s", month_end, total.to_dict())
    return total


def compute_dynamic_portfolio_returns(
    panel_6: pd.DataFrame, final_weights: pd.DataFrame
) -> pd.Series:
    """
    Compute dynamic portfolio returns from final stock weights.

    final_weights: DataFrame[month_end, ticker, weight]
    """
    df = panel_6[["month_end", "ticker", "forward_1m_return"]].copy()
    merged = df.merge(final_weights, on=["month_end", "ticker"], how="inner")
    merged["contrib"] = merged["weight"] * merged["forward_1m_return"]
    returns = merged.groupby("month_end")["contrib"].sum().sort_index()
    return returns

