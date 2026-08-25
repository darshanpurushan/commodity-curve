import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.storage_regression import (
    fit_convenience_yield_vs_storage,
    storage_implied_fair_convenience_yield,
)
from src.strategy.signals import generate_zscore_signal


def make_synthetic_storage_relationship(n=200, seed=42):
    """
    Generate synthetic data with a KNOWN inverse relationship between
    storage and convenience yield, so we can check the regression recovers
    the correct sign of the relationship (not the exact literature coefficients).
    """
    rng = np.random.default_rng(seed)
    storage = rng.uniform(300_000, 500_000, n)  # thousand barrels, realistic-ish range
    true_cy = 0.5 - 0.0000008 * storage + rng.normal(0, 0.01, n)  # decreasing in storage
    dates = pd.date_range("2020-01-01", periods=n, freq="W-FRI")
    return pd.DataFrame({
        "date": dates,
        "storage_kbbl": storage,
        "implied_convenience_yield": true_cy,
    })


def test_storage_regression_recovers_negative_relationship():
    df = make_synthetic_storage_relationship()
    fit = fit_convenience_yield_vs_storage(df)
    assert fit["coef_linear"] < 0  # more storage -> lower convenience yield
    assert fit["r_squared"] > 0.5  # should fit reasonably well on clean synthetic data
    assert fit["n_obs"] == len(df)


def test_storage_implied_prediction_direction():
    df = make_synthetic_storage_relationship()
    fit = fit_convenience_yield_vs_storage(df)
    low_storage_pred = storage_implied_fair_convenience_yield(fit, 300_000)
    high_storage_pred = storage_implied_fair_convenience_yield(fit, 500_000)
    assert low_storage_pred > high_storage_pred


def test_zscore_signal_generates_valid_positions():
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="W-FRI")
    rng = np.random.default_rng(1)
    mispricing = np.sin(np.linspace(0, 12, n)) * 0.05 + rng.normal(0, 0.005, n)
    df = pd.DataFrame({"date": dates, "mispricing": mispricing})

    out = generate_zscore_signal(df, lookback=20, entry_z=1.0, exit_z=0.25)
    assert set(out["position"].unique()).issubset({-1.0, 0.0, 1.0})
    assert len(out) == n
