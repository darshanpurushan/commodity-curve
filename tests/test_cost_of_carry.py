import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.cost_of_carry import implied_convenience_yield, curve_snapshot_to_carry


def test_implied_convenience_yield_zero_carry_case():
    # If F = S * exp((r+u)*T), convenience yield should come out ~0
    r, u, T = 0.05, 0.02, 0.5
    spot = 80.0
    futures_price = spot * np.exp((r + u) * T)
    y = implied_convenience_yield(spot, futures_price, T, r, u)
    assert abs(y) < 1e-8


def test_implied_convenience_yield_backwardation_positive_y():
    # Backwardation (futures < spot) should imply a positive convenience yield
    # when r+u is small
    y = implied_convenience_yield(
        spot=80.0, futures_price=78.0, time_to_expiry_years=0.25,
        risk_free_rate=0.04, storage_cost_rate=0.01,
    )
    assert y > 0


def test_curve_snapshot_to_carry_contango_flagged():
    # Rising prices with maturity = contango
    df = pd.DataFrame({
        "months_to_expiry": [1, 2, 3, 4],
        "price": [80.0, 80.5, 81.0, 81.5],
    })
    result = curve_snapshot_to_carry(
        df, spot=79.5, risk_free_rate=0.04, storage_cost_rate=0.01
    )
    assert "implied_convenience_yield" in result.columns
    assert "annualized_roll_yield" in result.columns
    assert (result["curve_state"].iloc[1:] == "contango_leg").all()


def test_curve_snapshot_to_carry_backwardation_flagged():
    df = pd.DataFrame({
        "months_to_expiry": [1, 2, 3, 4],
        "price": [82.0, 81.0, 80.0, 79.0],
    })
    result = curve_snapshot_to_carry(
        df, spot=83.0, risk_free_rate=0.04, storage_cost_rate=0.01
    )
    assert (result["curve_state"].iloc[1:] == "backwardation_leg").all()
