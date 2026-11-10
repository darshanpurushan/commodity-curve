import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import backtest_calendar_spread


def panel(positions):
    return pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=4),
        'price': [100., 101., 91., 93.],
        'far_price': [100.] * 4,
        'position': positions,
    })


@pytest.mark.parametrize('bad_position', [np.nan, np.inf, -np.inf, None])
def test_invalid_position_cannot_silently_erase_or_corrupt_pnl(bad_position):
    with pytest.raises(ValueError, match='finite numeric positions'):
        backtest_calendar_spread(panel([1., bad_position, 1., 0.]))


def test_fractional_exposure_retains_lagged_losses_and_turnover_costs():
    out = backtest_calendar_spread(panel([.5, -.25, 0., 0.]), transaction_cost_bps=10.)
    assert out['position_lagged'].tolist() == [0., .5, -.25, 0.]
    assert out['gross_pnl'].iloc[1:].tolist() == pytest.approx([.005, .025, 0.])
    assert out['transaction_cost'].tolist() == pytest.approx([0., .0005, .00075, .00025])
    assert out['net_pnl'].iloc[1:].tolist() == pytest.approx([.0045, .02425, -.00025])
