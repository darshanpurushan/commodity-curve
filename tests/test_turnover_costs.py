import pandas as pd
import pytest

from src.backtest.engine import backtest_calendar_spread


def test_reversal_costs_twice_as_much_as_entry_or_exit():
    data = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5),
        "price": [80.] * 5, "far_price": [81.] * 5,
        "position": [1., -1., 0., 0., 0.],
    })
    out = backtest_calendar_spread(data, transaction_cost_bps=10)
    assert out["transaction_cost"].tolist() == pytest.approx([0., .001, .002, .001, 0.])
    assert out["net_pnl"].sum() == pytest.approx(-.004)


def test_fractional_position_costs_follow_notional_change():
    data = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=4),
        "price": [80.] * 4, "far_price": [81.] * 4,
        "position": [.25, .75, .75, 0.],
    })
    out = backtest_calendar_spread(data, transaction_cost_bps=10)
    assert out["transaction_cost"].tolist() == pytest.approx([0., .00025, .0005, 0.])
