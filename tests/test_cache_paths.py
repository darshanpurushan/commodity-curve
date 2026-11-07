from unittest.mock import Mock

import pandas as pd
import pytest

from src.data import eia_client


@pytest.mark.parametrize("cache_path", ["storage.csv", "nested/data/storage.csv"])
def test_cache_creation_supports_bare_and_nested_paths(tmp_path, monkeypatch, cache_path):
    monkeypatch.chdir(tmp_path)
    expected = pd.DataFrame({"date": pd.to_datetime(["2025-01-03"]),
                             "storage_kbbl": [400000]})
    client = Mock()
    client.get_crude_storage.return_value = expected
    factory = Mock(return_value=client)
    monkeypatch.setattr(eia_client, "EIAClient", factory)
    actual = eia_client.load_storage_data_cached("2025-01-01", "2025-02-01", cache_path)
    pd.testing.assert_frame_equal(actual, expected)
    assert (tmp_path / cache_path).is_file()
    eia_client.load_storage_data_cached("2025-01-01", "2025-02-01", cache_path)
    factory.assert_called_once()
