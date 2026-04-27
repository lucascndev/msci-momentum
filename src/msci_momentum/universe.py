"""Universe loaders. v1: S&P 500 from Wikipedia, with on-disk cache."""

from __future__ import annotations

import io
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

_UA = "Mozilla/5.0 (compatible; msci-momentum/0.1; +https://github.com/lucascndev/msci-momentum)"

CACHE_DIR = Path(os.environ.get("MOMENTUM_CACHE", Path.home() / ".cache" / "msci-momentum"))
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass(frozen=True)
class UniverseMember:
    ticker: str          # yfinance-compatible symbol (e.g., BRK-B)
    raw_ticker: str      # source symbol (e.g., BRK.B)
    name: str
    country: str         # ISO country code or MSCI country name
    currency: str        # ISO currency code


def _to_yf_symbol(sym: str) -> str:
    return sym.replace(".", "-").strip()


def load_sp500(force_refresh: bool = False) -> list[UniverseMember]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "sp500.csv"
    if cache.exists() and not force_refresh:
        df = pd.read_csv(cache)
    else:
        req = urllib.request.Request(SP500_URL, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        df = tables[0]
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"Symbol": "raw_ticker", "Security": "name"})
        df = df[["raw_ticker", "name"]].copy()
        df["ticker"] = df["raw_ticker"].map(_to_yf_symbol)
        df["country"] = "USA"
        df["currency"] = "USD"
        df.to_csv(cache, index=False)

    return [
        UniverseMember(
            ticker=row.ticker,
            raw_ticker=row.raw_ticker,
            name=row.name,
            country=row.country,
            currency=row.currency,
        )
        for row in df.itertuples(index=False)
    ]


def load_universe(name: str, force_refresh: bool = False) -> list[UniverseMember]:
    if name.lower() in {"sp500", "s&p500", "snp500"}:
        return load_sp500(force_refresh=force_refresh)
    raise ValueError(f"Unknown universe: {name!r} (supported: sp500)")


def tickers(members: Iterable[UniverseMember]) -> list[str]:
    return [m.ticker for m in members]
