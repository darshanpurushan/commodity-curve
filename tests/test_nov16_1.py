import pytest
from test_backtest_engine import make_synthetic_backtest_input
from src.backtest.engine import backtest_calendar_spread
@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_cost(cost):
    with pytest.raises(ValueError, match="transaction_cost"):
        backtest_calendar_spread(make_synthetic_backtest_input(), transaction_cost_bps=cost)
def test_zero_cost():
    out = backtest_calendar_spread(make_synthetic_backtest_input(), transaction_cost_bps=0)
    assert out.transaction_cost.eq(0).all()
