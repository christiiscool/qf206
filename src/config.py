from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PathsConfig:
    project_root: Path
    data_raw: Path
    data_processed: Path
    outputs: Path


@dataclass
class DataConfig:
    index_symbol: str = "SPY"
    start_date: str = "2017-01-01"  # extra buffer before 2018 features
    end_date: str = "2025-12-31"
    price_column: str = "Adj Close"
    calendar: str = "BM"  # business month end
    min_history_months: int = 13  # ensure we can compute 12-1 momentum


@dataclass
class FeatureConfig:
    mom_12_1_window: int = 12
    mom_6_1_window: int = 6
    rev_1m_window: int = 1
    low_risk_lookback_days_short: int = 63  # ~3m
    low_risk_lookback_days_long: int = 252  # ~12m
    liquidity_lookback_days: int = 21  # ~1m
    amihud_lookback_days: int = 21
    overreaction_lookback_days: int = 5
    turnover_spike_lookback_days: int = 60
    window_52w_days: int = 252


@dataclass
class ModelConfig:
    regression_alpha: float = 0.5
    regression_l1_ratio: float = 0.5
    regression_max_iter: int = 5000

    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 300
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    use_mlp: bool = False
    mlp_hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])
    mlp_dropout: float = 0.2
    mlp_lr: float = 1e-3
    mlp_batch_size: int = 512
    mlp_max_epochs: int = 50
    mlp_patience: int = 5
    random_state: int = 42


@dataclass
class BacktestConfig:
    train_start: str = "2010-01-01"
    test_start: str = "2018-01-31"  # first month-end in test window
    rebalance_freq: str = "M"
    # Kept for compatibility, no longer used in factor allocation backtest.
    long_short_top_decile: float = 0.9
    long_short_bottom_decile: float = 0.1
    long_only_quintile: float = 0.8
    max_weight_long_only: float = 0.05
    sector_cap_long_only: float = 0.25
    transaction_cost_bps: float = 10.0  # one-way cost in basis points (Part 3)


@dataclass
class OptionsRiskConfig:
    enabled: bool = True
    options_data_path: Optional[Path] = Path(r"C:\Users\chris\Downloads\hhgzhvclobjfzv9d_csv.zip")
    ticker: str = "SPY"
    options_snapshot_freq: str = "M"
    min_dte: int = 20
    max_dte: int = 60
    rolling_lookback_months: int = 24
    warning_z_threshold: float = 1.0
    use_rolling_percentile_threshold: bool = True
    warning_percentile: float = 0.90
    put_target_delta_abs: float = 0.25
    put_target_dte_days: int = 30
    fixed_hedge_budget_pct_nav: float = 0.005
    under_hedge_budget_pct_nav: float = 0.0025
    over_hedge_budget_pct_nav: float = 0.0100
    hedge_sweep_budget_pct_nav: List[float] = field(default_factory=lambda: [0.0025, 0.0050, 0.0075, 0.0100, 0.0150])
    score_scaled_mild_percentile: float = 0.75
    score_scaled_high_percentile: float = 0.90
    score_scaled_extreme_percentile: float = 0.97
    score_scaled_mild_budget_pct_nav: float = 0.0025
    score_scaled_high_budget_pct_nav: float = 0.0050
    score_scaled_extreme_budget_pct_nav: float = 0.0100
    require_market_confirmation: bool = True
    market_drawdown_trigger: float = -0.08
    market_vol_percentile: float = 0.50
    state_change_cost_bps: float = 2.0
    chunksize: int = 250_000


@dataclass
class FactorTimingConfig:
    factors: List[str] = field(
        default_factory=lambda: ["momentum", "reversal", "lowvol", "behavioural"]
    )
    allow_short_factors: bool = True
    max_factor_weight: float = 0.6
    regularization: float = 1.0
    model_name: str = "elasticnet"  # Options: "ridge", "elasticnet", "xgboost"
    # Allocation smoothing (Part 5 - Option A)
    allocation_smoothing: float = 0.7  # gamma: w_t = gamma*w_target + (1-gamma)*w_{t-1}
    # Volatility targeting (Part 5 - Option B)
    enable_vol_targeting: bool = False
    target_vol: float = 0.12  # 12% annualized
    vol_lookback: int = 12  # months for realized vol estimation


@dataclass
class PipelineConfig:
    paths: PathsConfig
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    options_risk: OptionsRiskConfig = field(default_factory=OptionsRiskConfig)
    factor_timing: FactorTimingConfig = field(default_factory=FactorTimingConfig)
    demo_tickers: List[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "UNH"]
    )



def _detect_paths() -> PathsConfig:
    project_root = Path(__file__).resolve().parents[1]
    data_raw = project_root / "data" / "raw"
    data_processed = project_root / "data" / "processed"
    outputs = project_root / "outputs"
    return PathsConfig(
        project_root=project_root,
        data_raw=data_raw,
        data_processed=data_processed,
        outputs=outputs,
    )



def get_default_config() -> PipelineConfig:
    paths = _detect_paths()
    return PipelineConfig(paths=paths)



def to_dict(cfg: PipelineConfig) -> Dict[str, object]:
    def _asdict(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, list):
            return [_asdict(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _asdict(v) for k, v in obj.items()}
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _asdict(getattr(obj, k)) for k in obj.__dataclass_fields__.keys()}
        return str(obj)

    return _asdict(cfg)

