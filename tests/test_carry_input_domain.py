import pytest

from src.models.cost_of_carry import implied_convenience_yield


@pytest.mark.parametrize("name,value", [
    ("spot", 0.), ("spot", -80.), ("spot", float("nan")),
    ("futures_price", 0.), ("futures_price", -80.),
    ("futures_price", float("inf")),
    ("time_to_expiry_years", float("nan")),
    ("time_to_expiry_years", float("inf")),
    ("risk_free_rate", float("inf")), ("storage_cost_rate", float("nan")),
])
def test_log_carry_formula_rejects_inputs_outside_its_domain(name, value):
    args = dict(spot=80., futures_price=82., time_to_expiry_years=.5,
                risk_free_rate=.04, storage_cost_rate=.01)
    args[name] = value
    with pytest.raises(ValueError):
        implied_convenience_yield(**args)


def test_negative_finite_financing_rates_are_supported():
    result = implied_convenience_yield(80., 80., .5, -.01, .02)
    assert result == pytest.approx(.01)
