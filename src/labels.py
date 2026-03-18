from __future__ import annotations

import pandas as pd

from .config import PipelineConfig
from .utils import get_logger


def add_forward_return_and_labels(
    cfg: PipelineConfig, feature_panel: pd.DataFrame
) -> pd.DataFrame:
    """
    Add forward one-month returns and top-quintile classification labels.
    """
    logger = get_logger()
    df = feature_panel.sort_values(["ticker", "month_end"]).copy()

    df["forward_1m_return"] = df.groupby("ticker")["ret_1m"].shift(-1)

    def top_quintile_flag(group: pd.DataFrame) -> pd.Series:
        if group["forward_1m_return"].notna().sum() < 5:
            return pd.Series(False, index=group.index)
        q80 = group["forward_1m_return"].quantile(0.8)
        return group["forward_1m_return"] >= q80

    df["top_quintile_flag"] = (
        df.groupby("month_end", group_keys=False)
        .apply(top_quintile_flag)
        .astype(bool)
    )

    min_date = pd.to_datetime(cfg.backtest.train_start)
    df = df[df["month_end"] >= min_date].copy()

    logger.info(
        "Monthly panel with labels: %d rows, %d columns",
        df.shape[0],
        df.shape[1],
    )
    return df

