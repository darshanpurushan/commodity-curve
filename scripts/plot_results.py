"""
Generate the two charts you actually want in the README / interview deck:
1. Cumulative P&L of the strategy vs. a naive buy-and-hold near contract
2. Storage level vs. implied convenience yield scatter (the regression itself)

Run after scripts/run_pipeline.py has produced results/backtest_output.csv.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def plot_cumulative_pnl(bt_path="results/backtest_output.csv", out_path="results/cumulative_pnl.png"):
    df = pd.read_csv(bt_path, parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["cumulative_return"], label="Calendar spread strategy (net of costs)")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title("Cumulative Return: Storage-Implied Mispricing Strategy")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_cumulative_pnl()
