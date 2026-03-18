"""
Quick test to verify model selection works correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from src.config import FactorTimingConfig
from src.factor_timing_model import train_factor_timing_model


def test_model_selection():
    """Test that all three models can be instantiated and trained."""
    print("Testing model selection...")
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 50
    n_features = 3
    n_factors = 4
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=["realized_vol_3m", "mom_6m", "drawdown"]
    )
    Y_train = pd.DataFrame(
        np.random.randn(n_samples, n_factors),
        columns=["momentum", "reversal", "lowvol", "behavioural"]
    )
    
    models_to_test = ["ridge", "elasticnet", "xgboost"]
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        cfg = FactorTimingConfig(model_name=model_name, regularization=1.0)
        
        try:
            model = train_factor_timing_model(cfg, X_train, Y_train)
            
            # Test prediction
            X_test = X_train.iloc[:5]
            predictions = model.predict(X_test.values)
            
            assert predictions.shape == (5, n_factors), \
                f"Expected shape (5, {n_factors}), got {predictions.shape}"
            
            print(f"✓ {model_name} trained and predicted successfully")
            print(f"  Prediction shape: {predictions.shape}")
            
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            raise
    
    print("\n" + "=" * 60)
    print("All models tested successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_model_selection()
