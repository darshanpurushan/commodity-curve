import pandas as pd
import pytest
from src.backtest.engine import backtest_calendar_spread

@pytest.mark.parametrize('column,value',[('price',float('nan')),('price',float('inf')),('far_price',float('nan')),('far_price',float('inf')),('far_price',0.),('far_price',-1.)])
def test_invalid_valuation_inputs_are_rejected(column,value):
    frame=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=3),'price':[80.]*3,'far_price':[81.]*3,'position':[0.,1.,0.]})
    frame.loc[1,column]=value
    with pytest.raises(ValueError,match='finite quotes|positive far'):
        backtest_calendar_spread(frame)

def test_finite_negative_near_prices_do_not_break_spread_arithmetic():
    frame=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=3),'price':[-2.,-1.,0.],'far_price':[10.]*3,'position':[1.,1.,0.]})
    assert backtest_calendar_spread(frame,transaction_cost_bps=0)['net_pnl'].iloc[1] == pytest.approx(.1)
