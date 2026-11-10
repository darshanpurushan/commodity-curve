"""
Simple event-driven-ish backtest for a calendar-spread strategy.

Deliberately realistic on cost assumptions -- this is the part that most
"beginner" quant projects skip, and the part interviewers actually probe.
We charge:
    - a per-trade transaction cost (bps of notional) on every position change
    - no leverage / margin modeling (documented as a known simplification)

The spread P&L is computed as the return on holding
    long near-month future, short far-month future
scaled by the signal's position (+1 / -1 / 0), using the near/far prices
already present in the merged curve panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_calendar_spread(
    signal_df: pd.DataFrame,
    near_price_col: str = "price",
    transaction_cost_bps: float = 2.0,
) -> pd.DataFrame:
    """
    NOTE: this assumes signal_df already isolates the front-month contract
    (as produced by storage_regression.merge_curve_and_storage with
    front_month_only=True) and that a matching far-month price series has
    been joined in as 'far_price'. See scripts/run_pipeline.py for the join.
    """
    df = signal_df.copy().sort_values("date").reset_index(drop=True)
    if "far_price" not in df.columns:
        raise ValueError(
            "signal_df must include a 'far_price' column (the offsetting leg "
            "of the calendar spread). See scripts/run_pipeline.py."
        )

    df["spread"] = df[near_price_col] - df["far_price"]
    df["spread_return"] = df["spread"].diff() / df["far_price"].shift(1)

    # Position lagged by one period: today's signal determines tomorrow's exposure
    df["position_lagged"] = df["position"].shift(1).fillna(0)

    df["gross_pnl"] = df["position_lagged"] * df["spread_return"]

    trade_flag = df["position_lagged"].diff().fillna(df["position_lagged"]).abs() > 0
    df["transaction_cost"] = np.where(trade_flag, transaction_cost_bps / 1e4, 0.0)

    df["net_pnl"] = df["gross_pnl"] - df["transaction_cost"]
    df["cumulative_pnl"] = df["net_pnl"].cumsum()
    df["cumulative_return"] = (1 + df["net_pnl"]).cumprod() - 1

    return df


def daily_performance_summary(backtest_df: pd.DataFrame) -> dict:
    """Daily fixed-pair workflow metrics, with 252 observations per year."""
    return performance_summary(backtest_df, periods_per_year=252)


def performance_summary(backtest_df: pd.DataFrame, periods_per_year: int = 52) -> dict:
    returns = backtest_df["net_pnl"].dropna()
    if len(returns) == 0 or returns.std() == 0:
        return {"error": "insufficient data or zero variance"}

    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    n_trades = (backtest_df["position_lagged"].diff().fillna(0).abs() > 0).sum()
    hit_rate = (returns[returns != 0] > 0).mean() if (returns != 0).any() else np.nan

    return {
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_trades": int(n_trades),
        "hit_rate": hit_rate,
        "total_periods": len(returns),
    }
