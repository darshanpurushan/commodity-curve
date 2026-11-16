import pandas as pd
import pytest
from src.models.cost_of_carry import curve_snapshot_to_carry
@pytest.mark.parametrize("months", [0,-1])
def test_expired_tenor(months):
    with pytest.raises(ValueError, match="positive"):
        curve_snapshot_to_carry(pd.DataFrame({"months_to_expiry":[months],"price":[100]}),100,0,0)
def test_valid_tenor():
    out=curve_snapshot_to_carry(pd.DataFrame({"months_to_expiry":[1],"price":[100]}),100,0,0)
    assert out.implied_convenience_yield.iloc[0] == 0
