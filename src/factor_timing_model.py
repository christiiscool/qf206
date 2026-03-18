from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from .config import FactorTimingConfig
from .utils import get_logger


def build_regime_features(spy_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly regime features from SPY daily OHLCV.

    Features:
      - realized_vol_3m: 63d rolling vol (annualised)
      - market_momentum_6m: 6m price momentum
      - drawdown: max drawdown based on daily returns
    """
    logger = get_logger()
    spy = spy_daily.sort_values("date").copy()
    spy["ret_d"] = spy["adj_close"].pct_change()

    # Realised vol (63 trading days ~ 3 months)
    spy["realized_vol_3m"] = (
        spy["ret_d"].rolling(window=63, min_periods=40).std() * np.sqrt(252.0)
    )

    # 6m momentum using monthly prices
    monthly_price = (
        spy.set_index("date")["adj_close"].resample("M").last().to_frame("price")
    )
    monthly_price["mom_6m"] = monthly_price["price"] / monthly_price["price"].shift(
        6
    ) - 1.0

    # Drawdown from daily series
    cum = (1.0 + spy["ret_d"].fillna(0.0)).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    spy["drawdown"] = dd

    # Convert daily features to month-end
    spy_m = (
        spy.set_index("date")[["realized_vol_3m", "drawdown"]]
        .resample("M")
        .last()
    )

    regime = spy_m.join(monthly_price[["mom_6m"]], how="inner")
    regime.index.name = "month_end"
    logger.info("Built regime features with %d months", len(regime))
    return regime


def build_factor_timing_dataset(
    factor_returns: pd.DataFrame, regime_features: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct (X, Y) for factor timing.

    X_t = regime features at month t
    Y_t = factor returns for the next-month holding period decided at t.

    Note: In this project, `factor_returns` is computed using `forward_1m_return`
    from the stock panel, so each row already corresponds to the return over
    \(t \rightarrow t+1\) for weights formed at month-end \(t\). Therefore we
    align X and Y on the same index without look-ahead.
    """
    logger = get_logger()
    # Align on common months
    idx = factor_returns.index.intersection(regime_features.index)
    X = regime_features.loc[idx].copy()
    Y = factor_returns.loc[idx].copy()

    mask = ~(X.isna().any(axis=1) | Y.isna().any(axis=1))
    X = X[mask]
    Y = Y[mask]

    logger.info("Factor timing dataset: %d samples, %d features", X.shape[0], X.shape[1])
    return X, Y


def train_factor_timing_model(
    cfg: FactorTimingConfig, X_train: pd.DataFrame, Y_train: pd.DataFrame
) -> MultiOutputRegressor:
    """
    Train a multi-output regression model for factor timing.
    
    Supports multiple model types:
    - ridge: Ridge regression with L2 regularization
    - elasticnet: ElasticNet with L1 + L2 regularization
    - xgboost: Gradient boosted trees
    """
    logger = get_logger()
    
    if cfg.model_name == "ridge":
        base_model = Ridge(alpha=cfg.regularization, random_state=42)
        logger.info("Training Ridge regression with alpha=%.3f", cfg.regularization)
    elif cfg.model_name == "elasticnet":
        base_model = ElasticNet(
            alpha=cfg.regularization,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42
        )
        logger.info("Training ElasticNet with alpha=%.3f, l1_ratio=0.5", cfg.regularization)
    elif cfg.model_name == "xgboost":
        base_model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        logger.info("Training XGBoost with n_estimators=200, max_depth=3")
    else:
        raise ValueError(
            f"Unknown model_name '{cfg.model_name}'. "
            f"Choose from: 'ridge', 'elasticnet', 'xgboost'"
        )
    
    model = MultiOutputRegressor(base_model)
    model.fit(X_train.values, Y_train.values)
    logger.info("Model training completed")
    return model


def predict_factor_returns(
    model: MultiOutputRegressor, X_test: pd.DataFrame, factor_names
) -> pd.DataFrame:
    """
    Predict factor returns for the next month.
    """
    preds = model.predict(X_test.values)
    # X_test may contain multiple rows; keep index alignment
    return pd.DataFrame(preds, index=X_test.index, columns=list(factor_names))


