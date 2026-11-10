import numpy as np
import pandas as pd

from scripts import train_test_split as validation
from src.models.storage_regression import fit_convenience_yield_vs_storage
from src.strategy.signals import compute_mispricing_series


def test_future_targets_cannot_change_training_residuals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = np.linspace(300000., 500000., 60)
    data = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=60),
                         "storage_kbbl": storage,
                         "implied_convenience_yield": .5 - storage * 8e-7 + .01 * np.sin(np.arange(60))})
    seen = []
    def fake_grid(train):
        seen.append(train.copy())
        return pd.DataFrame([dict(lookback=13, entry_z=1., exit_z=.25, sharpe_ratio=.5)])
    monkeypatch.setattr(validation, "run_grid", fake_grid)
    monkeypatch.setattr(validation, "run_single_config", lambda *args: {"error": "offline fixture"})
    for future_shift in (0., 10.):
        changed = data.copy()
        changed.loc[42:, "implied_convenience_yield"] += future_shift
        full_fit = fit_convenience_yield_vs_storage(changed)
        full_residuals = compute_mispricing_series(changed, full_fit)
        monkeypatch.setattr(validation, "prepare_data", lambda: (full_residuals, full_fit))
        validation.main()
    pd.testing.assert_frame_equal(seen[0], seen[1])
    expected = compute_mispricing_series(data.iloc[:42], fit_convenience_yield_vs_storage(data.iloc[:42]))
    np.testing.assert_allclose(seen[0]["fair_convenience_yield"], expected["fair_convenience_yield"])
