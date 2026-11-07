import numpy as np
import pandas as pd

from scripts import train_test_split as validation
from scripts import tune_thresholds as tuning


def flat_data():
    return pd.DataFrame({"date": pd.date_range("2025-01-01", periods=20),
                         "near_price": [80.] * 20, "far_price": [81.] * 20,
                         "mispricing": [0.] * 20,
                         "storage_kbbl": np.linspace(300000., 400000., 20),
                         "implied_convenience_yield": np.linspace(.2, .1, 20)})


def small_grid(monkeypatch, module):
    monkeypatch.setattr(module, "LOOKBACKS", [4])
    monkeypatch.setattr(module, "ENTRY_ZS", [1.])
    monkeypatch.setattr(module, "EXIT_ZS", [.25])


def test_grid_without_trades_returns_empty_ranked_table(monkeypatch):
    small_grid(monkeypatch, validation)
    out = validation.run_grid(flat_data())
    assert out.empty
    assert "sharpe_ratio" in out.columns


def test_grid_excludes_undefined_sharpe_candidates(monkeypatch):
    small_grid(monkeypatch, validation)
    monkeypatch.setattr(validation, "performance_summary", lambda *args: dict(
        sharpe_ratio=np.nan, annualized_return=0., max_drawdown=0., n_trades=0))
    assert validation.run_grid(flat_data()).empty


def test_empty_searches_report_no_winner(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for module in (tuning, validation):
        small_grid(monkeypatch, module)
        monkeypatch.setattr(module, "prepare_data", lambda: (flat_data(), {"r_squared": 0., "n_obs": 20}))
        monkeypatch.setattr(module, "performance_summary", lambda *args: {"error": "insufficient data"})
        module.main()
        assert "No eligible" in capsys.readouterr().out
