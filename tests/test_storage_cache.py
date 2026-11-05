import pandas as pd

from src.data.eia_client import load_storage_data_cached


def test_cached_storage_is_limited_to_requested_date_window(tmp_path):
    path = tmp_path / "storage.csv"
    pd.DataFrame({"date": ["2025-01-17", "2025-01-03", "2025-01-10"],
                  "storage_kbbl": [420000, 400000, 410000]}).to_csv(path, index=False)
    out = load_storage_data_cached("2025-01-03", "2025-01-10", str(path))
    assert out["date"].tolist() == list(pd.to_datetime(["2025-01-03", "2025-01-10"]))
    assert out["storage_kbbl"].tolist() == [400000, 410000]
    assert out.index.tolist() == [0, 1]


def test_cached_window_without_observations_is_empty(tmp_path):
    path = tmp_path / "storage.csv"
    path.write_text("date,storage_kbbl\n2025-01-03,400000\n")
    out = load_storage_data_cached("2025-02-01", "2025-02-28", str(path))
    assert out.empty
