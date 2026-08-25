import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest.engine import backtest_calendar_spread, performance_summary


def make_synthetic_backtest_input(n=104, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="W-FRI")
    near = 80 + np.cumsum(rng.normal(0, 0.5, n))
    far = near - 1.0 + np.cumsum(rng.normal(0, 0.1, n))  # persistent small spread
    position = rng.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2])
    return pd.DataFrame({
        "date": dates, "price": near, "far_price": far, "position": position,
    })


def test_backtest_runs_and_produces_expected_columns():
    df = make_synthetic_backtest_input()
    result = backtest_calendar_spread(df, transaction_cost_bps=2.0)
    for col in ["gross_pnl", "transaction_cost", "net_pnl", "cumulative_pnl"]:
        assert col in result.columns


def test_transaction_costs_reduce_pnl_relative_to_gross():
    df = make_synthetic_backtest_input()
    result = backtest_calendar_spread(df, transaction_cost_bps=50.0)  # high cost
    assert result["net_pnl"].sum() <= result["gross_pnl"].sum()


def test_zero_position_strategy_has_zero_pnl():
    df = make_synthetic_backtest_input()
    df["position"] = 0.0
    result = backtest_calendar_spread(df, transaction_cost_bps=2.0)
    assert np.allclose(result["net_pnl"].fillna(0), 0.0)


def test_performance_summary_keys():
    df = make_synthetic_backtest_input()
    result = backtest_calendar_spread(df, transaction_cost_bps=2.0)
    summary = performance_summary(result)
    for key in ["annualized_return", "annualized_vol", "sharpe_ratio", "max_drawdown", "n_trades"]:
        assert key in summary
