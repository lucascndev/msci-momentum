"""Local short-term risk-free rates (MSCI methodology Appendix IV).

v1 supports USD only via the 13-week T-Bill yield (^IRX). The MSCI methodology
specifies per-country rates (Appendix IV); when we extend the universe beyond
the S&P 500 we add to ``RATE_TICKERS``.
"""

from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf

# Maps MSCI country name -> yfinance ticker for the local short-term rate.
# These tickers report the *annualized* yield in percent.
RATE_TICKERS: dict[str, str] = {
    "USA": "^IRX",  # 13-week US T-Bill yield, annualized %
}


def annualized_rate(country: str, as_of: pd.Timestamp) -> float:
    """Annualized risk-free rate (decimal, e.g., 0.045 = 4.5%) at ``as_of``.

    Falls back to the most recent observation on or before the requested date.
    """
    if country not in RATE_TICKERS:
        raise ValueError(
            f"No risk-free rate configured for {country!r}. "
            f"Supported: {sorted(RATE_TICKERS)}"
        )
    sym = RATE_TICKERS[country]
    end = pd.Timestamp(as_of) + pd.Timedelta(days=7)
    start = end - pd.DateOffset(months=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            sym,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            actions=False,
        )
    if df is None or df.empty:
        raise RuntimeError(f"No rate data for {sym} near {as_of.date()}")
    closes = df["Close"].dropna()
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    cutoff = closes.loc[closes.index <= pd.Timestamp(as_of)]
    if cutoff.empty:
        cutoff = closes
    return float(cutoff.iloc[-1]) / 100.0


def horizon_rate(country: str, as_of: pd.Timestamp, months: int) -> float:
    """Subtractable rate over ``months`` months: annualized * months/12."""
    return annualized_rate(country, as_of) * (months / 12.0)
