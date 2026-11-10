"""
Train/test split validation of the tuned z-score thresholds.

The grid search in tune_thresholds.py finds the BEST-looking thresholds on
the full dataset -- which is guaranteed to look good, since it was chosen
specifically because it looked good on that data. This script answers the
real question: do those thresholds still work on data they were NOT chosen
using?

Method:
    1. Split the data chronologically into a TRAIN period (first ~70%) and
       a TEST period (last ~30%). Chronological, not random -- random
       shuffling would leak future information into the training set,
       which isn't a mistake you can make when trading in real time.
    2. Run the same grid search, but only on the TRAIN period.
    3. Take the best TRAIN-period thresholds and run them, unmodified, on
       the TEST period.
    4. Report both Sharpe ratios side by side. A large drop from train to
       test Sharpe is a warning sign of overfitting; a modest, consistent
       Sharpe in both periods is a much stronger result to report.

Usage:
    export EIA_API_KEY=your_key_here
    python scripts/train_test_split.py
"""
from __future__ import annotations

import sys
import os
import itertools
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.tune_thresholds import (
    prepare_data, LOOKBACKS, ENTRY_ZS, EXIT_ZS, rank_grid_results,
)
from src.strategy.signals import generate_zscore_signal
from src.strategy.signals import compute_mispricing_series
from src.models.storage_regression import fit_convenience_yield_vs_storage
from src.backtest.engine import backtest_calendar_spread, performance_summary

TRAIN_FRACTION = 0.7


def run_grid(data: pd.DataFrame) -> pd.DataFrame:
    results = []
    for lookback, entry_z, exit_z in itertools.product(LOOKBACKS, ENTRY_ZS, EXIT_ZS):
        if exit_z >= entry_z:
            continue
        signal_df = generate_zscore_signal(data, lookback=lookback, entry_z=entry_z, exit_z=exit_z)
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
    return rank_grid_results(results)


def run_single_config(data: pd.DataFrame, lookback: int, entry_z: float, exit_z: float) -> dict:
    signal_df = generate_zscore_signal(data, lookback=lookback, entry_z=entry_z, exit_z=exit_z)
    bt_input = signal_df.rename(columns={"near_price": "price"})
    bt = backtest_calendar_spread(bt_input, transaction_cost_bps=2.0)
    return performance_summary(bt)


def main():
    print("Preparing full dataset...")
    merged, fit = prepare_data()
    merged = merged.sort_values("date").reset_index(drop=True)

    split_idx = int(len(merged) * TRAIN_FRACTION)
    train = merged.iloc[:split_idx].reset_index(drop=True)
    test = merged.iloc[split_idx:].reset_index(drop=True)
    # prepare_data's full-sample fit is suitable for in-sample exploration,
    # but validation must recompute both residual series from training only.
    train_fit = fit_convenience_yield_vs_storage(train)
    train = compute_mispricing_series(train, train_fit)
    test = compute_mispricing_series(test, train_fit)
    print(f"Train period: {train['date'].min().date()} to {train['date'].max().date()} ({len(train)} rows)")
    print(f"Test period:  {test['date'].min().date()} to {test['date'].max().date()} ({len(test)} rows)")

    print("\nRunning grid search on TRAIN period only...")
    train_results = run_grid(train)
    if train_results.empty:
        print("No eligible training configurations with a defined Sharpe ratio; test evaluation skipped.")
        return
    best = train_results.iloc[0]
    print(f"Best train-period config: lookback={int(best['lookback'])}, "
          f"entry_z={best['entry_z']}, exit_z={best['exit_z']}, "
          f"train Sharpe={best['sharpe_ratio']:.3f}")

    print("\nApplying that SAME config, unmodified, to the TEST period...")
    test_summary = run_single_config(
        test, int(best["lookback"]), best["entry_z"], best["exit_z"]
    )

    print("\n--- Train vs. Test comparison ---")
    print(f"  Train Sharpe: {best['sharpe_ratio']:.3f}")
    if "error" in test_summary:
        print(f"  Test Sharpe:  could not compute ({test_summary['error']}) "
              f"-- likely too few trades in the test window to evaluate.")
    else:
        print(f"  Test Sharpe:  {test_summary['sharpe_ratio']:.3f}")
        print(f"  Test annualized return: {test_summary['annualized_return']:.3f}")
        print(f"  Test max drawdown: {test_summary['max_drawdown']:.3f}")
        print(f"  Test n_trades: {test_summary['n_trades']}")

    os.makedirs("results", exist_ok=True)
    train_results.to_csv("results/train_grid_search.csv", index=False)
    print("\nFull train-period grid saved to results/train_grid_search.csv")
    print(
        "\nHonest read: if test Sharpe is meaningfully lower than train Sharpe, "
        "that gap IS the overfitting -- report it, don't hide it. A strategy "
        "that holds up reasonably well out-of-sample, even if less impressive "
        "than the in-sample number, is a far more credible result."
    )


if __name__ == "__main__":
    main()
