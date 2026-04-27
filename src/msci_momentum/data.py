"""Price and market-cap fetching via yfinance, with on-disk CSV cache.

We only need three things to compute MSCI Momentum:
  * Monthly closes for the 13 months prior to the rebalance date
    (split-adjusted, dividends NOT reinvested — MSCI uses local price returns).
  * Weekly closes for the 3 years prior to the rebalance date (for volatility).
  * A snapshot of free-float market capitalization at the rebalance date.

yfinance gives us monthly/weekly prices easily. For market cap, yfinance exposes
``shares_outstanding`` per Ticker.info; we use total mcap = shares * price as a
practical proxy for the free-float mcap MSCI actually uses.
"""

from __future__ import annotations

import hashlib
import logging
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from msci_momentum.universe import CACHE_DIR

log = logging.getLogger(__name__)


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _ticker_hash(tickers: list[str]) -> str:
    h = hashlib.md5("|".join(sorted(tickers)).encode()).hexdigest()
    return h[:10]


def _download(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers=tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # group_by="ticker" → outer level is ticker, inner is OHLCV.
        closes = {}
        for tk in tickers:
            if tk in raw.columns.get_level_values(0):
                series = raw[tk].get("Close")
                if series is not None:
                    closes[tk] = series
        df = pd.DataFrame(closes)
    else:
        df = raw[["Close"]].rename(columns={"Close": tickers[0]})

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def fetch_monthly_closes(
    tickers: list[str],
    rebalance_date: pd.Timestamp,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Month-end closes covering >= 14 months prior to rebalance_date."""
    end = pd.Timestamp(rebalance_date) + pd.offsets.MonthEnd(1)
    start = end - pd.DateOffset(months=18)
    cache = _cache_path(f"monthly_{end.date()}_{_ticker_hash(tickers)}.csv")
    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    df = _download(tickers, start, end, "1mo")
    if not df.empty:
        df.to_csv(cache)
    return df


def fetch_weekly_closes(
    tickers: list[str],
    rebalance_date: pd.Timestamp,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Weekly closes covering >= 3 years prior to rebalance_date."""
    end = pd.Timestamp(rebalance_date) + pd.Timedelta(days=7)
    start = end - pd.DateOffset(years=3, days=14)
    cache = _cache_path(f"weekly_{end.date()}_{_ticker_hash(tickers)}.csv")
    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    df = _download(tickers, start, end, "1wk")
    if not df.empty:
        df.to_csv(cache)
    return df


def fetch_security_metadata(
    tickers: list[str],
    rebalance_date: pd.Timestamp,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Per-ticker static metadata: shares, float shares, sector.

    Returns a DataFrame indexed by ticker with columns:
      shares_outstanding, float_shares, sector

    yfinance's ``Ticker.info`` is rate-limited; we cache once per rebalance date.
    Cache file is keyed by date + ticker hash.
    """
    cache = _cache_path(
        f"meta_{pd.Timestamp(rebalance_date).date()}_{_ticker_hash(tickers)}.csv"
    )
    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0)

    rows: dict[str, dict] = {}
    failed: list[str] = []
    for i, tk in enumerate(tickers, 1):
        info: dict | None = None
        for attempt in range(3):
            try:
                info = yf.Ticker(tk).get_info()
                break
            except Exception as e:  # noqa: BLE001
                log.debug("info attempt %d failed for %s: %s", attempt + 1, tk, e)
                time.sleep(0.5 * (attempt + 1))
        if info:
            so = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            fs = info.get("floatShares")
            sec = info.get("sector")
            rows[tk] = {
                "shares_outstanding": float(so) if so else np.nan,
                "float_shares": float(fs) if fs else np.nan,
                "sector": sec if sec else "Unknown",
            }
        else:
            failed.append(tk)
        if i % 50 == 0:
            log.info("metadata: %d/%d fetched", i, len(tickers))
        time.sleep(0.05)
    if failed:
        log.warning("metadata unavailable for %d tickers: %s",
                    len(failed), failed[:10])
    meta = pd.DataFrame.from_dict(rows, orient="index")
    if not meta.empty:
        meta.index.name = "ticker"
        meta.to_csv(cache)
    return meta


def fetch_market_caps(
    tickers: list[str],
    rebalance_date: pd.Timestamp,
    monthly_closes: pd.DataFrame,
    use_cache: bool = True,
    use_float: bool = True,
) -> pd.Series:
    """Market cap at rebalance date.

    With ``use_float=True`` (default), uses ``float_shares`` × price as a
    free-float-mcap approximation, matching MSCI's float-weighted mcap intent.
    Falls back to ``shares_outstanding`` × price for names where yfinance
    doesn't return ``floatShares`` (typical: a few percent of a 500-name run).

    With ``use_float=False`` we use full shares outstanding (legacy v1
    behavior — kept so tests and ad-hoc inspection can compare).
    """
    meta = fetch_security_metadata(tickers, rebalance_date, use_cache=use_cache)
    if meta.empty:
        raise ValueError("No metadata returned for any ticker")

    if use_float:
        # Prefer float_shares; fall back to shares_outstanding when null.
        shares = meta["float_shares"].where(
            meta["float_shares"].notna() & (meta["float_shares"] > 0),
            meta["shares_outstanding"],
        )
    else:
        shares = meta["shares_outstanding"]

    rb = pd.Timestamp(rebalance_date)
    monthly = monthly_closes.loc[monthly_closes.index <= rb]
    if monthly.empty:
        raise ValueError(f"No monthly prices on/before {rb.date()}")
    last_price = monthly.iloc[-1]

    common = shares.dropna().index.intersection(last_price.index)
    mcap = (shares.loc[common] * last_price.loc[common]).dropna()
    mcap.name = "market_cap"
    return mcap


def month_end_closes_at_offset(
    monthly: pd.DataFrame, rebalance_date: pd.Timestamp, months_back: int
) -> pd.Series:
    """Return the close that is `months_back` month-ends before rebalance_date.

    MSCI uses P_{T-1}, P_{T-7}, P_{T-13}: the month-end closes 1, 7, and 13
    full months before T. We index into the month-end series by integer offset
    from the latest month-end at or before T.
    """
    rb = pd.Timestamp(rebalance_date)
    monthly = monthly.sort_index()
    cutoff = monthly.loc[monthly.index <= rb]
    if cutoff.empty:
        raise ValueError(f"No monthly bars at or before {rb.date()}")
    idx = len(cutoff) - 1 - months_back
    if idx < 0:
        # Not enough history for this lag; return all-NaN row so missing data
        # propagates downstream rather than crashing.
        return pd.Series(np.nan, index=monthly.columns, name=f"P_T-{months_back}")
    row = cutoff.iloc[idx].copy()
    row.name = f"P_T-{months_back}"
    return row
