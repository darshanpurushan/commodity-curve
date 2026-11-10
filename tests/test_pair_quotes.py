import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.futures_data import fetch_fixed_pair_history


def fake_yahoo(monkeypatch, near, far):
    dates = pd.date_range("2025-01-02", periods=len(near))
    def ticker(symbol):
        values = near if symbol == "CL=F" else far
        return SimpleNamespace(history=lambda **kwargs: pd.DataFrame({"Close": values}, index=dates))
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=ticker))


def test_pair_history_drops_missing_and_infinite_quotes(monkeypatch):
    fake_yahoo(monkeypatch, [80., np.nan, 82., np.inf], [81., 82., np.nan, 84.])
    out = fetch_fixed_pair_history("2025-01-01", "2025-02-01")
    assert out["near_price"].tolist() == [80.]
    assert out["far_price"].tolist() == [81.]


def test_pair_history_rejects_history_without_valid_paired_quotes(monkeypatch):
    fake_yahoo(monkeypatch, [np.nan, 80.], [81., np.nan])
    with pytest.raises(RuntimeError, match="No overlapping"):
        fetch_fixed_pair_history("2025-01-01", "2025-02-01")
