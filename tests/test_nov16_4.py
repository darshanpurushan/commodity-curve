import numpy as np
import pytest
from src.models.cost_of_carry import implied_convenience_yield
@pytest.mark.parametrize("spot,future", [(1e-300,1e300),(1e300,1e-300)])
def test_extreme_finite_ratio(spot,future):
    value=implied_convenience_yield(spot,future,1,0,0)
    assert np.isfinite(value)
    assert value == pytest.approx(np.log(spot)-np.log(future))
