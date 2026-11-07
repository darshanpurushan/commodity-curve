"""
Grid search over z-score entry/exit thresholds and the rolling lookback
window, to find the combination with the best in-sample Sharpe ratio.

This is Extension #1 from the README: the original 1.0/0.25 thresholds were
a reasonable starting guess, not fit to data. This script actually fits them.

NOTE: this is an IN-SAMPLE search -- it will always find a threshold
combination that looks better on the data it was tuned on. For a genuinely
credible result, the next step (not in this script) would be to split the
data into a training period (fit thresholds here) and a held-out test
period (check the tuned thresholds still work there). Report both numbers
honestly if you take that next step -- in-sample-only results are known to
overstate real-world performance.

Usage:
    export EIA_API_KEY=your_key_here
    python scripts/tune_thresholds.py
"""
from __future__ import annotations

import sys
import os
import datetime as dt
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.eia_client import load_storage_data_cached
from src.data.futures_data import fetch_fixed_pair_history
from src.models.cost_of_carry import implied_convenience_yield
from src.models.storage_regression import fit_convenience_yield_vs_storage
from src.strategy.signals import generate_zscore_signal
from src.backtest.engine import backtest_calendar_spread, performance_summary

START_DATE = (dt.date.today() - dt.timedelta(days=450)).isoformat()
END_DATE = dt.date.today().isoformat()
RISK_FREE_RATE = 0.045
STORAGE_COST_RATE = 0.015
FAR_MONTHS_AHEAD = 6

# The grid: every combination of these gets backtested
LOOKBACKS = [13, 26, 39, 52]        # weeks-equivalent in trading days-ish; adjust as you like
ENTRY_ZS = [0.5, 0.75, 1.0, 1.25, 1.5]
EXIT_ZS = [0.1, 0.25, 0.5]


def rank_grid_results(results):
    """Return a stable table of candidates with a defined Sharpe ratio."""
    columns = ["lookback", "entry_z", "exit_z", "sharpe_ratio",
               "annualized_return", "max_drawdown", "n_trades"]
    frame = pd.DataFrame(results, columns=columns)
    frame["sharpe_ratio"] = pd.to_numeric(frame["sharpe_ratio"], errors="coerce")
    return frame.loc[np.isfinite(frame["sharpe_ratio"])].sort_values(
        "sharpe_ratio", ascending=False)


def prepare_data():
    pair_hist = fetch_fixed_pair_history(START_DATE, END_DATE, far_months_ahead=FAR_MONTHS_AHEAD)
    storage_df = load_storage_data_cached(START_DATE, END_DATE)

    pair_hist = pair_hist[pair_hist["months_to_expiry"] > 0].copy()
    pair_hist["implied_convenience_yield"] = pair_hist.apply(
        lambda row: implied_convenience_yield(
            spot=row["near_price"], futures_price=row["far_price"],
            time_to_expiry_years=max(row["months_to_expiry"] / 12.0, 1e-6),
            risk_free_rate=RISK_FREE_RATE, storage_cost_rate=STORAGE_COST_RATE,
        ), axis=1,
    )

    pair_hist["date"] = pd.to_datetime(pair_hist["date"]).astype("datetime64[ns]")
    storage_sorted = storage_df.sort_values("date").copy()
    storage_sorted["date"] = pd.to_datetime(storage_sorted["date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(pair_hist.sort_values("date"), storage_sorted, on="date", direction="backward")
    merged = merged.dropna(subset=["storage_kbbl"]).reset_index(drop=True)

    fit = fit_convenience_yield_vs_storage(merged)
    merged["fair_convenience_yield"] = fit["model"].predict(
        merged[["storage_kbbl"]].assign(storage_kbbl_sq=merged["storage_kbbl"] ** 2)
        [["storage_kbbl", "storage_kbbl_sq"]]
    )
    merged["mispricing"] = merged["implied_convenience_yield"] - merged["fair_convenience_yield"]
    return merged, fit


def main():
    print("Preparing data (this hits live APIs once, then reuses the result for every grid cell)...")
    merged, fit = prepare_data()
    print(f"  R^2 = {fit['r_squared']:.3f}, n_obs = {fit['n_obs']}")

    results = []
    combos = list(itertools.product(LOOKBACKS, ENTRY_ZS, EXIT_ZS))
    print(f"Running grid search over {len(combos)} combinations...")

    for lookback, entry_z, exit_z in combos:
        if exit_z >= entry_z:
            continue  # exit band must be tighter than entry band, or it never trades
        signal_df = generate_zscore_signal(merged, lookback=lookback, entry_z=entry_z, exit_z=exit_z)
        bt_input = signal_df.rename(columns={"near_price": "price"})
        bt = backtest_calendar_spread(bt_input, transaction_cost_bps=2.0)
        summary = performance_summary(bt)
        if "error" in summary:
            continue
        results.append({
            "lookback": lookback, "entry_z": entry_z, "exit_z": exit_z,
            "sharpe_ratio": summary["sharpe_ratio"],
            "annualized_return": summary["annualized_return"],
            "max_drawdown": summary["max_drawdown"],
            "n_trades": summary["n_trades"],
        })

    results_df = rank_grid_results(results)
    if results_df.empty:
        print("No eligible configurations with a defined Sharpe ratio; no winner selected.")
        return
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/threshold_grid_search.csv", index=False)

    print("\nTop 5 combinations by in-sample Sharpe ratio:")
    print(results_df.head(5).to_string(index=False))
    print("\nFull grid saved to results/threshold_grid_search.csv")
    print(
        "\nReminder: this is an IN-SAMPLE result. The top combination is "
        "guaranteed to look good on this exact data -- that doesn't mean "
        "it will hold up going forward. Treat this as a starting point for "
        "further validation, not a finished answer."
    )


if __name__ == "__main__":
    main()
