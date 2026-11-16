import pandas as pd
import pytest
from src.strategy.signals import generate_zscore_signal
@pytest.mark.parametrize("kwargs", [{"lookback":1},{"entry_z":-1},{"exit_z":0},{"exit_z":2},{"entry_z":float("nan")},{"exit_z":float("inf")}])
def test_invalid_configuration(kwargs):
    df=pd.DataFrame({"date":pd.date_range("2020-01-01", periods=4),"mispricing":[1,2,3,1]})
    with pytest.raises(ValueError):
        generate_zscore_signal(df,**kwargs)
