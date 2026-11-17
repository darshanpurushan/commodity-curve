import pytest
import pandas as pd
from test_backtest_engine import make_synthetic_backtest_input
from src.backtest.engine import backtest_calendar_spread
@pytest.mark.parametrize("missing", [True, False])
def test_ambiguous_dates(missing):
    df = make_synthetic_backtest_input()
    df.loc[1, "date"] = pd.NaT if missing else df.loc[0, "date"]
    with pytest.raises(ValueError, match="dates"):
        backtest_calendar_spread(df)
def test_unsorted_unique_dates():
    df = make_synthetic_backtest_input()
    pd.testing.assert_frame_equal(backtest_calendar_spread(df), backtest_calendar_spread(df.iloc[::-1]))
