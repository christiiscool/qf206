from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import requests
import yfinance as yf

from .config import PipelineConfig
from .utils import get_logger


WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass
class Universe:
    tickers: List[str]
    metadata: pd.DataFrame


def _sp500_constituents_path(cfg: PipelineConfig) -> Path:
    return cfg.paths.data_raw / "sp500_constituents.csv"


def download_sp500_constituents(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Retrieve S&P 500 tickers and cache to CSV.

    Primary source: yfinance.tickers_sp500()
    Fallback: Wikipedia table (with User-Agent) if needed.
    """
    logger = get_logger()
    path = _sp500_constituents_path(cfg)

    # 1) Try yfinance helper first (no HTML parsing in our code).
    try:
        logger.info("Retrieving S&P 500 tickers via yfinance.tickers_sp500()")
        tickers = yf.tickers_sp500()
        if isinstance(tickers, (list, tuple)) and len(tickers) > 0:
            table = pd.DataFrame({"ticker": sorted(set(tickers))})
            table["download_timestamp_utc"] = dt.datetime.utcnow()
            table["source"] = "yfinance.tickers_sp500"

            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)
            logger.info("Saved S&P 500 universe (yfinance) to %s", path)
            return table
        logger.warning("yfinance.tickers_sp500() returned an empty result, falling back")
    except Exception as exc:  # noqa: BLE001
        logger.error("yfinance.tickers_sp500() failed: %s. Falling back to Wikipedia.", exc)

    # 2) Fallback to Wikipedia HTML table with explicit User-Agent.
    logger.info("Downloading S&P 500 constituents from %s", WIKI_SP500_URL)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        table = tables[0]
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to download S&P 500 constituents from Wikipedia: %s", exc)
        if path.exists():
            logger.warning("Falling back to cached %s", path)
            return pd.read_csv(path)
        raise RuntimeError(
            "Unable to retrieve S&P 500 constituents via yfinance or Wikipedia. "
            "If this persists, manually download the ticker list and save it to "
            f"{path} with at least a 'ticker' column."
        ) from exc

    table.columns = [c.lower().strip().replace(" ", "_") for c in table.columns]
    table.rename(columns={"symbol": "ticker"}, inplace=True)

    table["download_timestamp_utc"] = dt.datetime.utcnow()
    table["source_url"] = WIKI_SP500_URL

    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Saved S&P 500 universe (Wikipedia) to %s", path)
    return table


def load_sp500_constituents(cfg: PipelineConfig, force_refresh: bool = False) -> pd.DataFrame:
    path = _sp500_constituents_path(cfg)
    if force_refresh or not path.exists():
        return download_sp500_constituents(cfg)
    return pd.read_csv(path)


def get_full_universe(cfg: PipelineConfig, force_refresh: bool = False) -> Universe:
    df = load_sp500_constituents(cfg, force_refresh=force_refresh)
    tickers = sorted(df["ticker"].unique().tolist())
    return Universe(tickers=tickers, metadata=df)


def get_demo_universe(cfg: PipelineConfig) -> Universe:
    """
    Fixed 6-stock subset used for demo plots and interpretability.
    """
    full = get_full_universe(cfg)
    demo_tickers = [t for t in cfg.demo_tickers if t in full.tickers]
    meta = full.metadata[full.metadata["ticker"].isin(demo_tickers)].copy()
    return Universe(tickers=demo_tickers, metadata=meta)

