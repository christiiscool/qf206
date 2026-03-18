from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from src.config import get_default_config  # noqa: E402
from src.data_download import download_ohlcv, load_daily_data  # noqa: E402
from src.features import build_monthly_feature_panel  # noqa: E402
from src.labels import add_forward_return_and_labels  # noqa: E402
from src.model_comparison import run_model_comparison  # noqa: E402
from src.plots import plot_demo_universe_panel  # noqa: E402
from src.utils import ensure_directories, get_logger  # noqa: E402


def main() -> None:
    cfg = get_default_config()
    logger = get_logger()

    ensure_directories(cfg.paths.data_raw, cfg.paths.data_processed, cfg.paths.outputs)

    # Universe is now fixed to the six demo tickers
    tickers = cfg.demo_tickers
    logger.info("Using 6-stock demo universe: %s", ", ".join(tickers))

    # Download or load data
    prices, spy = download_ohlcv(cfg, tickers)

    # Build features and labels
    feature_panel = build_monthly_feature_panel(cfg, prices, spy)
    labeled_panel = add_forward_return_and_labels(cfg, feature_panel)

    panel_path = cfg.paths.data_processed / "monthly_panel.parquet"
    labeled_panel.to_parquet(panel_path, index=False)
    logger.info("Saved monthly panel to %s", panel_path)

    # Run model comparison
    models_to_test = ["ridge", "elasticnet", "xgboost"]
    comparison_df = run_model_comparison(cfg, labeled_panel, spy, models_to_test)

    # Generate universe plot (only once)
    plot_demo_universe_panel(cfg, labeled_panel, cfg.paths.outputs)

    logger.info("=" * 60)
    logger.info("Model comparison pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
