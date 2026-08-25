"""
Cost-of-carry model for commodity futures.

Theory
------
For a storable commodity, no-arbitrage links the futures price to the spot
price through the cost of carry:

    F(t,T) = S(t) * exp[(r + u - y) * (T - t)]

where:
    S(t)  = spot price at time t
    r     = risk-free / financing rate
    u     = storage cost (per unit time, as a fraction of value)
    y     = convenience yield (the benefit of holding physical inventory
             rather than a paper claim on future delivery -- e.g. being able
             to meet unexpected demand, avoid stockouts, keep a refinery running)
    T - t = time to delivery, in years

Rearranged, we can back out the *implied convenience yield* from any two
points on the observed futures curve:

    y = r + u - ln(F(t,T) / S(t)) / (T - t)

This is the same logic as backing out a forward rate from a bond curve --
just replace "financing rate" with "convenience yield" and "coupon" with
"storage cost." If you've built or used a yield curve stripper before, this
will feel familiar.

Market convention: when convenience yield is high relative to r+u, the curve
is in BACKWARDATION (near-term price > far-term price) -- physical scarcity,
strong near-term demand, low inventories. When convenience yield is low, the
curve is in CONTANGO -- ample inventories, storage is cheap relative to
waiting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_convenience_yield(
    spot: float,
    futures_price: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    storage_cost_rate: float,
) -> float:
    """
    Back out the annualized convenience yield implied by a single futures
    price relative to spot.
    """
    if time_to_expiry_years <= 0:
        raise ValueError("time_to_expiry_years must be positive")
    log_ratio = np.log(futures_price / spot)
    return risk_free_rate + storage_cost_rate - log_ratio / time_to_expiry_years


def curve_snapshot_to_carry(
    curve_df: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    storage_cost_rate: float,
) -> pd.DataFrame:
    """
    Given a single-date curve snapshot (columns: months_to_expiry, price),
    compute implied convenience yield and annualized roll yield at each tenor.

    Roll yield approximates the return an investor earns just from holding a
    futures position as it "rolls down" (or up) the curve toward spot as
    expiry approaches -- this is the P&L driver in a calendar-spread strategy.
    """
    out = curve_df.copy()
    out["time_to_expiry_years"] = out["months_to_expiry"] / 12.0

    out["implied_convenience_yield"] = out.apply(
        lambda row: implied_convenience_yield(
            spot=spot,
            futures_price=row["price"],
            time_to_expiry_years=max(row["time_to_expiry_years"], 1e-6),
            risk_free_rate=risk_free_rate,
            storage_cost_rate=storage_cost_rate,
        ),
        axis=1,
    )

    # Annualized "roll yield" between consecutive contracts on the curve:
    # ln(near/far) / (time_far - time_near), i.e. the slope of the curve.
    out = out.sort_values("months_to_expiry").reset_index(drop=True)
    out["annualized_roll_yield"] = np.nan
    for i in range(1, len(out)):
        near, far = out.iloc[i - 1], out.iloc[i]
        dt_years = far["time_to_expiry_years"] - near["time_to_expiry_years"]
        if dt_years > 0:
            out.loc[i, "annualized_roll_yield"] = (
                np.log(near["price"] / far["price"]) / dt_years
            )

    out["curve_state"] = np.where(
        out["price"].diff().fillna(0) < 0, "backwardation_leg", "contango_leg"
    )
    return out
