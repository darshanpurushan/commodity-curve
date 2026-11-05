"""
Storage / convenience-yield regression.

Core hypothesis (well-established in commodity finance, going back to
Working (1949) and Fama & French (1987, 1988)): convenience yield is a
*decreasing, convex* function of inventory levels. When inventories are low,
the marginal unit of physical stock is precious (stockout risk), so
convenience yield spikes and the curve backwardates. When inventories are
abundant, convenience yield collapses toward zero (or storage costs
dominate), and the curve contangoes.

This module regresses the implied convenience yield (or, as a simpler proxy,
the front-month curve slope) against inventory levels to test that
relationship empirically on real EIA storage data, and to generate a
mean-reversion-style trading signal: when inventories imply the curve
"should" be more backwardated/contangoed than it currently is, that's a
potential trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def merge_curve_and_storage(
    carry_panel: pd.DataFrame,
    storage_df: pd.DataFrame,
    front_month_only: bool = True,
) -> pd.DataFrame:
    """
    Join daily/weekly curve-carry data with weekly EIA storage data on date
    (as-of / nearest prior storage report, since storage is reported weekly
    on a lag).

    Parameters
    ----------
    carry_panel : output of cost_of_carry.curve_snapshot_to_carry, concatenated
        across dates, with a 'date' column.
    storage_df : output of eia_client, columns ['date', 'storage_kbbl'].
    """
    df = carry_panel.copy()
    if front_month_only:
        # Exclude the spot-proxy contract itself (months_to_expiry == 0) so we
        # don't compute a "convenience yield" from comparing spot to spot.
        df = df[df["months_to_expiry"] > 0]
        # Keep an actual front-contract row, including its missing values.
        # groupby.first() selects each column's first non-null value and
        # can silently splice another maturity's yield into this row.
        df = df.sort_values(["date", "months_to_expiry"]).drop_duplicates("date", keep="first")

    storage_sorted = storage_df.sort_values("date")
    merged = pd.merge_asof(
        df.sort_values("date"),
        storage_sorted,
        on="date",
        direction="backward",  # use the most recent storage report as-of each date
    )
    return merged


def fit_convenience_yield_vs_storage(merged_df: pd.DataFrame) -> dict:
    """
    Fit: implied_convenience_yield ~ storage_kbbl (+ storage_kbbl^2 for convexity)

    Returns dict with fitted model, coefficients, and R^2, so you can report
    "a 1% decrease in inventories is associated with an X bps increase in
    convenience yield" -- the kind of relationship a trading desk actually
    cares about.
    """
    data = merged_df.dropna(subset=["implied_convenience_yield", "storage_kbbl"]).copy()
    data["storage_kbbl_sq"] = data["storage_kbbl"] ** 2

    X = data[["storage_kbbl", "storage_kbbl_sq"]].values
    y = data["implied_convenience_yield"].values

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    return {
        "model": model,
        "coef_linear": model.coef_[0],
        "coef_quadratic": model.coef_[1],
        "intercept": model.intercept_,
        "r_squared": r2,
        "n_obs": len(data),
    }


def storage_implied_fair_convenience_yield(fit_result: dict, storage_level: float) -> float:
    """Predict the 'fair' convenience yield for a given inventory level."""
    model = fit_result["model"]
    return float(model.predict([[storage_level, storage_level ** 2]])[0])
