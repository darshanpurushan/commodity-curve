import pandas as pd
import pytest

from src.backtest.engine import performance_summary


@pytest.mark.parametrize("returns, expected", [
    ([-.2, .1], -.2),
    ([-.2, -.1], -.28),
    ([-.2, .5, -.1], -.2),
    ([.1, -.2], -.2),
    ([.01, .02], 0.),
])
def test_summary_measures_drawdown_from_initial_capital(returns, expected):
    frame = pd.DataFrame({"net_pnl": returns, "position_lagged": [1.] * len(returns)})
    summary = performance_summary(frame)
    assert summary["max_drawdown"] == pytest.approx(expected)
