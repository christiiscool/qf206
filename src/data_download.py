from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .config import PipelineConfig
from .utils import get_logger, save_json


def _prices_path(cfg: PipelineConfig) -> str:
    return str(cfg.paths.data_raw / "daily_ohlcv.parquet")


def _spy_path(cfg: PipelineConfig) -> str:
    return str(cfg.paths.data_raw / "daily_spy.parquet")


def _sources_metadata_path(cfg: PipelineConfig) -> str:
    return str(cfg.paths.data_raw / "data_sources.json")


def download_ohlcv(
    cfg: PipelineConfig, tickers: Iterable[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download daily OHLCV for the given tickers and SPY index proxy.

    Returns
    -------
    prices : DataFrame
        Long-form daily OHLCV with columns:
        ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume'].
    spy : DataFrame
        Daily OHLCV for the index symbol.
    """
    logger = get_logger()
    all_tickers = sorted(set(tickers))
    if cfg.data.index_symbol not in all_tickers:
        all_tickers.append(cfg.data.index_symbol)

    logger.info("Downloading daily OHLCV for %d tickers via yfinance", len(all_tickers))
    data = yf.download(
        tickers=all_tickers,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for ticker in all_tickers:
            if ticker not in data.columns.get_level_values(0):
                continue
            df_t = data[ticker].copy()
            df_t.columns = [c.lower().replace(" ", "_") for c in df_t.columns]
            df_t["ticker"] = ticker
            frames.append(df_t)
        prices = pd.concat(frames, axis=0)
    else:
        prices = data.copy()
        prices["ticker"] = cfg.data.index_symbol

    prices.reset_index(inplace=True)
    prices.rename(columns={"Date": "date"}, inplace=True)
    prices["date"] = pd.to_datetime(prices["date"])
    prices.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adj close": "adj_close",
            "adj_close": "adj_close",
            "volume": "volume",
        },
        inplace=True,
    )

    spy = prices[prices["ticker"] == cfg.data.index_symbol].copy()
    assets = prices[prices["ticker"] != cfg.data.index_symbol].copy()

    logger.info("Saving daily OHLCV to %s", _prices_path(cfg))
    assets.to_parquet(_prices_path(cfg), index=False)
    logger.info("Saving SPY OHLCV to %s", _spy_path(cfg))
    spy.to_parquet(_spy_path(cfg), index=False)

    save_json(
        {
            "prices_source": "yfinance.download",
            "index_symbol": cfg.data.index_symbol,
            "start_date": cfg.data.start_date,
            "end_date": cfg.data.end_date,
            "tickers_count": int(len(all_tickers)),
        },
        cfg.paths.data_raw / "data_sources.json",
    )

    return assets, spy


def load_daily_data(cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_parquet(_prices_path(cfg))
    spy = pd.read_parquet(_spy_path(cfg))
    prices["date"] = pd.to_datetime(prices["date"])
    spy["date"] = pd.to_datetime(spy["date"])
    return prices, spy

