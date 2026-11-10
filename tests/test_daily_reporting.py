import numpy as np
import pandas as pd
import pytest
from scripts import run_pipeline, tune_thresholds, train_test_split

@pytest.mark.parametrize('module',[run_pipeline,tune_thresholds,train_test_split])
def test_daily_workflows_annualize_daily_returns(module):
    frame=pd.DataFrame({'net_pnl':[.01,-.005,.02], 'position_lagged':[0,1,1]})
    summary=module.performance_summary(frame)
    assert summary['annualized_return'] == pytest.approx(frame['net_pnl'].mean()*252)
    assert summary['annualized_vol'] == pytest.approx(frame['net_pnl'].std()*np.sqrt(252))
