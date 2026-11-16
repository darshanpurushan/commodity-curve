import pandas as pd
from src.backtest.engine import performance_summary
def test_single_return_is_insufficient():
    assert "error" in performance_summary(pd.DataFrame({"net_pnl": [0.1], "position_lagged": [1]}))
def test_two_returns_have_metrics():
    assert "sharpe_ratio" in performance_summary(pd.DataFrame({"net_pnl": [0.1, -0.1], "position_lagged": [0,1]}))
