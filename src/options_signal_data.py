from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import OptionsRiskConfig
from .utils import get_logger


DEFAULT_OPTION_FIELDS = [
    "date",
    "exdate",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "delta",
    "optionid",
    "ticker",
    "symbol",
    "forward_price",
]

UNDERLYING_CANDIDATE_COLUMNS = [
    "underlying_price",
    "spot",
    "spot_price",
    "forward_price",
]


def standardize_options_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: col.strip().lower() for col in df.columns}
    return df.rename(columns=rename_map)


def _detect_underlying_column(columns: Iterable[str]) -> Optional[str]:
    normalized = {str(col).strip().lower() for col in columns}
    for candidate in UNDERLYING_CANDIDATE_COLUMNS:
        if candidate in normalized:
            return candidate
    return None


def _clean_options_chunk(chunk: pd.DataFrame, ticker: str) -> pd.DataFrame:
    chunk = standardize_options_columns(chunk)

    if "ticker" in chunk.columns:
        chunk = chunk[chunk["ticker"].astype(str).str.upper() == ticker.upper()].copy()

    if chunk.empty:
        return chunk

    for col in ["date", "exdate"]:
        if col in chunk.columns:
            chunk[col] = pd.to_datetime(chunk[col], errors="coerce")

    numeric_cols = [
        "strike_price",
        "best_bid",
        "best_offer",
        "volume",
        "open_interest",
        "impl_volatility",
        "delta",
        "forward_price",
    ]
    for col in numeric_cols:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    if "strike_price" in chunk.columns:
        strike_median = chunk["strike_price"].dropna().median()
        if pd.notna(strike_median) and strike_median > 1000:
            chunk["strike_price"] = chunk["strike_price"] / 1000.0

    chunk["cp_flag"] = chunk["cp_flag"].astype(str).str.upper().str.strip()
    chunk["dte"] = (chunk["exdate"] - chunk["date"]).dt.days
    chunk["mid"] = (chunk["best_bid"] + chunk["best_offer"]) / 2.0

    underlying_col = _detect_underlying_column(chunk.columns)
    if underlying_col is not None:
        chunk["underlying_price"] = pd.to_numeric(chunk[underlying_col], errors="coerce")
    else:
        chunk["underlying_price"] = np.nan

    keep_cols = [
        "date",
        "exdate",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
        "mid",
        "volume",
        "open_interest",
        "impl_volatility",
        "delta",
        "underlying_price",
        "optionid",
        "ticker",
        "symbol",
        "dte",
    ]
    present_cols = [col for col in keep_cols if col in chunk.columns]
    chunk = chunk[present_cols].copy()

    chunk = chunk.dropna(subset=["date", "exdate", "cp_flag", "strike_price"])
    chunk = chunk[chunk["cp_flag"].isin(["C", "P"])]
    chunk = chunk[chunk["dte"] >= 0]
    chunk = chunk[(chunk["best_bid"].fillna(-1) >= 0) & (chunk["best_offer"].fillna(-1) >= 0)]
    chunk = chunk[chunk["best_offer"].fillna(-1) >= chunk["best_bid"].fillna(-1)]
    chunk = chunk[chunk["mid"].fillna(-1) >= 0]
    chunk = chunk[chunk["open_interest"].fillna(0) >= 0]
    chunk = chunk[chunk["volume"].fillna(0) >= 0]

    dedupe_cols = [col for col in ["date", "exdate", "cp_flag", "strike_price", "optionid"] if col in chunk.columns]
    chunk = chunk.drop_duplicates(subset=dedupe_cols, keep="last")
    return chunk


def load_wrds_options_export(path: Path | str, cfg: OptionsRiskConfig) -> pd.DataFrame:
    logger = get_logger()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Options file not found: {path}")

    reader = pd.read_csv(
        path,
        compression="infer",
        usecols=lambda col: str(col).strip().lower() in set(DEFAULT_OPTION_FIELDS),
        chunksize=cfg.chunksize,
        low_memory=True,
    )

    cleaned_chunks = []
    for chunk in reader:
        cleaned = _clean_options_chunk(chunk, ticker=cfg.ticker)
        if not cleaned.empty:
            cleaned_chunks.append(cleaned)

    if not cleaned_chunks:
        raise ValueError(f"No usable option rows found for ticker {cfg.ticker} in {path}")

    # Avoid a full-frame global sort here because the WRDS export is large and
    # the downstream monthly builders either group explicitly or sort the
    # smaller working slices they need.
    df = pd.concat(cleaned_chunks, ignore_index=True)

    logger.info("Loaded %d cleaned %s option rows from %s", len(df), cfg.ticker, path)
    return df
