"""
End-to-end pipeline: fetch data -> fit carry model -> fit storage regression
-> generate signal -> backtest -> report.

Usage:
    export EIA_API_KEY=your_key_here
    pip install -r requirements.txt
    python scripts/run_pipeline.py

This script hits live APIs (EIA + Yahoo Finance via yfinance) and will take
a few minutes to run depending on the date range. For development/iteration,
run the unit tests in tests/ instead -- they use synthetic data and run in
under 3 seconds.
"""
from __future__ import annotations

import datetime as dt
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.eia_client import load_storage_data_cached
from src.data.futures_data import fetch_fixed_pair_history
from src.models.cost_of_carry import implied_convenience_yield
from src.models.storage_regression import (
    fit_convenience_yield_vs_storage,
)
from src.strategy.signals import generate_zscore_signal
from src.backtest.engine import backtest_calendar_spread, daily_performance_summary as performance_summary

# ---- Config ----
# NOTE: Yahoo Finance's free data for individual WTI contract-month tickers
# (e.g. CLZ25.NYM) is reliable mainly for currently/recently-active contracts.
# The fixed-pair approach (one continuous front-month series + one specific
# far contract, each pulled as a single full-history call) sidesteps most of
# that coverage problem -- see src/data/futures_data.fetch_fixed_pair_history.
START_DATE = (dt.date.today() - dt.timedelta(days=450)).isoformat()
END_DATE = dt.date.today().isoformat()
RISK_FREE_RATE = 0.045          # approximate; swap in real SOFR/T-bill data for production use
STORAGE_COST_RATE = 0.015       # rough estimate; tankage lease rates vary by region/contract
FAR_MONTHS_AHEAD = 6


def main():
    print(f"Fetching WTI near/far contract history {START_DATE} -> {END_DATE} ...")
    pair_hist = fetch_fixed_pair_history(START_DATE, END_DATE, far_months_ahead=FAR_MONTHS_AHEAD)
    print(f"  Rows fetched: {len(pair_hist)} (far leg: {pair_hist['far_ticker'].iloc[0]})")

    print("Fetching EIA crude storage data ...")
    storage_df = load_storage_data_cached(START_DATE, END_DATE)

    print("Computing implied convenience yield per date ...")
    pair_hist = pair_hist[pair_hist["months_to_expiry"] > 0].copy()
    pair_hist["implied_convenience_yield"] = pair_hist.apply(
        lambda row: implied_convenience_yield(
            spot=row["near_price"],
            futures_price=row["far_price"],
            time_to_expiry_years=max(row["months_to_expiry"] / 12.0, 1e-6),
            risk_free_rate=RISK_FREE_RATE,
            storage_cost_rate=STORAGE_COST_RATE,
        ),
        axis=1,
    )

    print("Merging curve data with storage levels ...")
    pair_hist["date"] = pd.to_datetime(pair_hist["date"]).astype("datetime64[ns]")
    storage_sorted = storage_df.sort_values("date").copy()
    storage_sorted["date"] = pd.to_datetime(storage_sorted["date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        pair_hist.sort_values("date"), storage_sorted, on="date", direction="backward",
    )

    n_merged_rows = len(merged.dropna(subset=["implied_convenience_yield", "storage_kbbl"]))
    print(f"  Rows with both curve + storage data after merge: {n_merged_rows}")
    if n_merged_rows < 30:
        print(
            "  WARNING: fewer than 30 observations. Any regression fit on this "
            "few points is not statistically meaningful -- treat R^2 and "
            "coefficients as a pipeline sanity check, not a real result. "
            "Consider a longer date range now that the fixed-pair fetch is "
            "far more data-efficient, or a paid data source for full rigor."
        )

    print("Fitting storage -> convenience yield regression ...")
    fit = fit_convenience_yield_vs_storage(merged)
    print(f"  R^2 = {fit['r_squared']:.3f}, n_obs = {fit['n_obs']}")
    print(f"  linear coef = {fit['coef_linear']:.3e}, quadratic coef = {fit['coef_quadratic']:.3e}")

    print("Generating mispricing signal ...")
    merged = merged.dropna(subset=["storage_kbbl"]).reset_index(drop=True)
    merged["fair_convenience_yield"] = fit["model"].predict(
        merged[["storage_kbbl"]].assign(storage_kbbl_sq=merged["storage_kbbl"] ** 2)
        [["storage_kbbl", "storage_kbbl_sq"]]
    )
    merged["mispricing"] = merged["implied_convenience_yield"] - merged["fair_convenience_yield"]
    signal_df = generate_zscore_signal(merged, lookback=26, entry_z=1.0, exit_z=0.25)

    print("Running backtest ...")
    bt_input = signal_df.rename(columns={"near_price": "price"})
    bt = backtest_calendar_spread(bt_input, transaction_cost_bps=2.0)
    summary = performance_summary(bt)

    print("\n--- Performance Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    os.makedirs("results", exist_ok=True)
    bt.to_csv("results/backtest_output.csv", index=False)
    print("\nSaved full backtest output to results/backtest_output.csv")


if __name__ == "__main__":
    main()
