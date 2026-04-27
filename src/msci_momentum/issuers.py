"""Maps tickers to issuer IDs so the issuer cap aggregates dual-class shares.

MSCI applies the issuer cap at the *issuer* level. yfinance has no concept of
issuer ID, so we hard-code the dual-class names that exist in the S&P 500.
A ticker not in the table maps to itself (single-class issuer).

If the universe expands beyond S&P 500, this table needs to grow — or be
replaced with an external issuer-mapping data source.
"""

from __future__ import annotations

ISSUER_GROUPS: dict[str, str] = {
    "GOOGL": "ALPHABET",
    "GOOG": "ALPHABET",
    "FOX": "FOX_CORP",
    "FOXA": "FOX_CORP",
    "NWS": "NEWS_CORP",
    "NWSA": "NEWS_CORP",
}


def issuer_id(ticker: str) -> str:
    """Issuer ID for ``ticker``; defaults to the ticker itself."""
    return ISSUER_GROUPS.get(ticker, ticker)
