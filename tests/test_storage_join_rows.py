import pandas as pd

from src.models.storage_regression import merge_curve_and_storage


def test_front_month_join_preserves_the_actual_contract_row():
    date = pd.Timestamp("2025-01-03")
    panel = pd.DataFrame({"date": [date, date], "months_to_expiry": [1, 2],
                          "price": [80., 82.],
                          "implied_convenience_yield": [float("nan"), .04]})
    storage = pd.DataFrame({"date": [date], "storage_kbbl": [400000.]})
    out = merge_curve_and_storage(panel, storage)
    assert len(out) == 1
    assert out.iloc[0]["months_to_expiry"] == 1
    assert out.iloc[0]["price"] == 80.
    assert pd.isna(out.iloc[0]["implied_convenience_yield"])


def test_storage_join_uses_prior_report_and_excludes_spot_proxy():
    dates = pd.to_datetime(["2025-01-03", "2025-01-10"])
    panel = pd.DataFrame({"date": [dates[0]] * 2 + [dates[1]],
                          "months_to_expiry": [0, 1, 1], "price": [79., 80., 81.]})
    storage = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-09"]),
                            "storage_kbbl": [400000., 410000.]})
    out = merge_curve_and_storage(panel, storage)
    assert out["storage_kbbl"].tolist() == [400000., 410000.]
    assert out["months_to_expiry"].tolist() == [1, 1]
