import numpy as np
import pandas as pd
import pytest

from src.models.cost_of_carry import curve_snapshot_to_carry


@pytest.mark.parametrize("near,far", [(1e300, 1e-300), (1e-300, 1e300), (100., 105.)])
def test_roll_yield_avoids_intermediate_ratio_overflow(near, far):
    curve = pd.DataFrame({"months_to_expiry": [1, 13], "price": [near, far]})
    result = curve_snapshot_to_carry(curve, spot=1., risk_free_rate=0., storage_cost_rate=0.)
    actual = result.loc[1, "annualized_roll_yield"]
    assert np.isfinite(actual)
    assert actual == pytest.approx(np.log(near) - np.log(far))
    assert pd.isna(result.loc[0, "annualized_roll_yield"])
