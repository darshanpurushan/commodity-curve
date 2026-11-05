"""
WTI crude oil (CL) futures curve construction.

Yahoo Finance publishes individual futures contracts under tickers like
'CLZ25.NYM' (CL = WTI crude, Z = December, 25 = 2025). We pull a handful of
contracts spanning the front of the curve and stitch them into a term
structure snapshot for each trading date.

Data source: yfinance (wraps Yahoo Finance, free, no API key).
    pip install yfinance
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

# NYMEX month codes
MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}


@dataclass
class ContractSpec:
    ticker: str          # e.g. "CLZ25.NYM"
    contract_month: dt.date  # first-of-month delivery date, e.g. 2025-12-01


def build_contract_ticker(root: str, year: int, month: int) -> str:
    code = MONTH_CODES[month]
    yy = str(year)[-2:]
    return f"{root}{code}{yy}.NYM"


def next_n_contract_months(as_of: dt.date, n: int) -> list[ContractSpec]:
    """Return the next n monthly contract specs starting from as_of's month."""
    specs = []
    y, m = as_of.year, as_of.month
    for _ in range(n):
        specs.append(ContractSpec(
            ticker=build_contract_ticker("CL", y, m),
            contract_month=dt.date(y, m, 1),
        ))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return specs


def fetch_curve_snapshot(as_of: dt.date, n_months: int = 12) -> pd.DataFrame:
    """
    Fetch a single-date snapshot of the WTI futures curve: price for each of
    the next `n_months` contract months, plus months-to-expiry.

    Requires yfinance. Network calls happen here -- run this locally, not in
    a sandboxed/offline environment.
    """
    import yfinance as yf  # local import so the rest of the package works without it installed

    specs = next_n_contract_months(as_of, n_months)
    rows = []
    for spec in specs:
        hist = yf.Ticker(spec.ticker).history(
            start=as_of - dt.timedelta(days=5),
            end=as_of + dt.timedelta(days=1),
        )
        if hist.empty:
            continue
        price = hist["Close"].iloc[-1]
        months_to_expiry = (spec.contract_month.year - as_of.year) * 12 + (
            spec.contract_month.month - as_of.month
        )
        rows.append({
            "ticker": spec.ticker,
            "contract_month": spec.contract_month,
            "months_to_expiry": months_to_expiry,
            "price": price,
        })

    if not rows:
        raise RuntimeError(
            "No contract data returned. Yahoo's futures ticker coverage can be "
            "spotty for far-dated contracts -- try a smaller n_months, or check "
            "that yfinance is installed and up to date."
        )
    return pd.DataFrame(rows).sort_values("months_to_expiry").reset_index(drop=True)


def fetch_curve_history(start: str, end: str, n_months: int = 8, freq: str = "W-FRI") -> pd.DataFrame:
    """
    DEPRECATED PATH: builds a panel of full curve snapshots by re-selecting
    a different set of contract tickers for every historical date. In
    practice, Yahoo Finance's free data does not reliably backfill price
    history for the many distinct expiry-specific tickers this requires, so
    this approach silently loses most of its sample. Kept for reference /
    for anyone with a paid data feed (CME DataMine, Nasdaq Data Link) where
    ticker coverage is complete. For the free-data path, use
    fetch_fixed_pair_history instead.
    """
    dates = pd.date_range(start=start, end=end, freq=freq)
    panels = []
    for d in dates:
        try:
            snap = fetch_curve_snapshot(d.date(), n_months=n_months)
            snap["date"] = d
            panels.append(snap)
        except RuntimeError:
            continue  # skip dates with no data (holidays, gaps)
    if not panels:
        raise RuntimeError("No curve history could be built for the given date range.")
    return pd.concat(panels, ignore_index=True)


def fetch_fixed_pair_history(
    start: str,
    end: str,
    far_months_ahead: int = 6,
    near_ticker: str = "CL=F",
) -> pd.DataFrame:
    """
    Reliable free-data alternative to fetch_curve_history.

    Instead of re-picking a new set of contract tickers for every historical
    date (which requires Yahoo to have backfilled dozens of thinly-tracked
    tickers), this tracks ONE fixed pair over the whole window:
      - near leg: CL=F, Yahoo's continuous front-month WTI series (deep,
        reliable daily history, since it's the most actively watched ticker)
      - far leg: one specific contract, `far_months_ahead` months out from
        `end` (chosen once, then its full history is pulled in a single call)

    This is also a more realistic trade construction: a calendar-spread
    strategy in practice tracks a specific spread over time rather than
    re-selecting which two contracts to compare every week.

    Returns a DataFrame: ['date', 'near_price', 'far_price', 'far_ticker',
    'months_to_expiry'] where months_to_expiry is computed per-date as the
    distance from that date to the far contract's delivery month (so it
    correctly decays as the far contract approaches expiry).

    Caveat: the far leg's price history only exists from whenever that
    contract was first listed. If `start` predates the far contract's
    listing, those early rows will simply be absent after the join --
    check the row count in the output rather than assuming full coverage.
    """
    import yfinance as yf

    end_date = pd.to_datetime(end).date()
    y, m = end_date.year, end_date.month
    m += far_months_ahead
    y += (m - 1) // 12
    m = ((m - 1) % 12) + 1
    far_spec = ContractSpec(
        ticker=build_contract_ticker("CL", y, m),
        contract_month=dt.date(y, m, 1),
    )

    near_hist = yf.Ticker(near_ticker).history(start=start, end=end)[["Close"]]
    near_hist = near_hist.rename(columns={"Close": "near_price"})
    near_hist.index = pd.to_datetime(near_hist.index.date)

    far_hist = yf.Ticker(far_spec.ticker).history(start=start, end=end)[["Close"]]
    if far_hist.empty:
        raise RuntimeError(
            f"No data returned for far-leg contract {far_spec.ticker}. It may not "
            f"have been listed yet over this date range -- try a smaller "
            f"far_months_ahead, or shift the date range closer to today."
        )
    far_hist = far_hist.rename(columns={"Close": "far_price"})
    far_hist.index = pd.to_datetime(far_hist.index.date)

    merged = near_hist.join(far_hist, how="inner").reset_index().rename(columns={"index": "date"})
    # An index overlap is insufficient when either contract lacks a valid
    # quote. Retain only paired finite prices, without filling missing data.
    valid_quotes = np.isfinite(merged[["near_price", "far_price"]]).all(axis=1)
    merged = merged.loc[valid_quotes].reset_index(drop=True)
    if merged.empty:
        raise RuntimeError(
            "No overlapping dates with valid quotes between near and far contract histories. "
            "Check that both tickers have data over the requested range."
        )

    merged["months_to_expiry"] = merged["date"].apply(
        lambda d: (far_spec.contract_month.year - d.year) * 12
        + (far_spec.contract_month.month - d.month)
    )
    merged["far_ticker"] = far_spec.ticker
    return merged[["date", "near_price", "far_price", "far_ticker", "months_to_expiry"]]
