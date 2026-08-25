"""
Signal generation: trade calendar spreads when the *observed* curve slope
deviates from what the storage regression says it "should" be.

Trade logic
-----------
- fair_convenience_yield = model prediction from current inventory level
- actual_convenience_yield = implied from the observed front-month curve
- mispricing = actual - fair

If mispricing is very negative (curve is more contangoed than inventories
justify), we expect convenience yield to mean-revert upward, i.e. the curve
should flatten/backwardate -> go LONG the calendar spread (long near
contract, short far contract).

If mispricing is very positive (curve is more backwardated than inventories
justify), expect convenience yield to fall -> go SHORT the calendar spread.

This is deliberately a simple z-score mean-reversion signal. The point of
this project is not to claim alpha -- it's to demonstrate the pipeline:
economic model -> empirical fit -> signal -> backtest -> honest evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_mispricing_series(
    merged_df: pd.DataFrame,
    fit_result: dict,
) -> pd.DataFrame:
    df = merged_df.copy()
    df["fair_convenience_yield"] = fit_result["model"].predict(
        df[["storage_kbbl"]].assign(storage_kbbl_sq=df["storage_kbbl"] ** 2)
        [["storage_kbbl", "storage_kbbl_sq"]]
    )
    df["mispricing"] = df["implied_convenience_yield"] - df["fair_convenience_yield"]
    return df


def generate_zscore_signal(
    df: pd.DataFrame,
    lookback: int = 26,
    entry_z: float = 1.0,
    exit_z: float = 0.25,
) -> pd.DataFrame:
    """
    Rolling z-score of the mispricing series -> discrete position signal.

    position:
        +1  = long calendar spread (mispricing very negative -> expect reversion up)
        -1  = short calendar spread (mispricing very positive -> expect reversion down)
         0  = flat (inside the exit band)
    """
    out = df.copy().sort_values("date").reset_index(drop=True)
    roll_mean = out["mispricing"].rolling(lookback, min_periods=lookback // 2).mean()
    roll_std = out["mispricing"].rolling(lookback, min_periods=lookback // 2).std()
    out["mispricing_z"] = (out["mispricing"] - roll_mean) / roll_std.replace(0, np.nan)

    position = np.zeros(len(out))
    current = 0
    for i, z in enumerate(out["mispricing_z"]):
        if np.isnan(z):
            position[i] = current
            continue
        if current == 0:
            if z < -entry_z:
                current = 1
            elif z > entry_z:
                current = -1
        else:
            if abs(z) < exit_z:
                current = 0
        position[i] = current

    out["position"] = position
    return out
