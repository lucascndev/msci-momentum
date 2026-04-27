"""End-to-end snapshot orchestration.

Single entry point used by the CLI, the Streamlit GUI, and the static-site
builder so they don't drift. Returns a ``Snapshot`` dataclass plus the raw
``scores`` DataFrame for callers that want to inspect non-eligible names.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from msci_momentum.data import (
    fetch_market_caps,
    fetch_monthly_closes,
    fetch_security_metadata,
)
from msci_momentum.issuers import ISSUER_GROUPS, issuer_id
from msci_momentum.momentum import build_inputs_for_universe, compute_momentum_scores
from msci_momentum.portfolio import build_portfolio
from msci_momentum.universe import load_universe, tickers as universe_tickers

# If fewer than this fraction of the universe ends up with a Momentum value,
# we fail loud rather than publish a degraded snapshot.
MIN_ELIGIBILITY_RATIO = 0.90


@dataclass
class Snapshot:
    date: pd.Timestamp
    generated_at: datetime
    params: dict
    universe_size: int
    eligible_size: int
    float_coverage: int
    portfolio: pd.DataFrame   # indexed by ticker; cols include sector, issuer
    scores: pd.DataFrame      # full scoring output (incl. non-selected names)
    z_winsorized: pd.Series   # cross-section, eligible only


def run_snapshot(
    rebalance_date: pd.Timestamp,
    *,
    universe_name: str = "sp500",
    top_n: int = 100,
    issuer_cap: float | None = 0.05,
    ad_hoc: bool = False,
    use_cache: bool = True,
    use_float: bool = True,
    min_eligibility_ratio: float = MIN_ELIGIBILITY_RATIO,
) -> Snapshot:
    rebalance = pd.Timestamp(rebalance_date)
    members = load_universe(universe_name)
    tks = universe_tickers(members)

    inputs = build_inputs_for_universe(
        tks, rebalance, country="USA", use_cache=use_cache
    )
    scores = compute_momentum_scores(inputs, use_only_6m=ad_hoc)

    monthly = fetch_monthly_closes(tks, rebalance, use_cache=use_cache)
    meta = fetch_security_metadata(tks, rebalance, use_cache=use_cache)
    mcap = fetch_market_caps(
        tks, rebalance, monthly, use_cache=use_cache, use_float=use_float
    )

    eligible = scores["combined"].dropna().index.intersection(mcap.index)
    eligibility_ratio = len(eligible) / max(len(tks), 1)
    if eligibility_ratio < min_eligibility_ratio:
        raise RuntimeError(
            f"Eligibility too low: {len(eligible)}/{len(tks)} "
            f"({eligibility_ratio:.1%} < {min_eligibility_ratio:.0%}). "
            "Likely partial yfinance fetch — refusing to publish."
        )

    scores_e = scores.loc[eligible]
    mcap_e = mcap.loc[eligible]

    portfolio = build_portfolio(
        scores_e,
        parent_mcap=mcap_e,
        n=top_n,
        issuer_cap=issuer_cap,
        issuer_map=ISSUER_GROUPS,
    )

    # Decorate portfolio with sector + issuer for downstream display.
    portfolio["sector"] = (
        meta["sector"].reindex(portfolio.index).fillna("Unknown")
        if "sector" in meta.columns
        else "Unknown"
    )
    portfolio["issuer"] = [issuer_id(t) for t in portfolio.index]

    float_coverage = int(
        meta["float_shares"].notna().sum() if "float_shares" in meta.columns else 0
    )

    return Snapshot(
        date=rebalance.normalize(),
        generated_at=datetime.now(timezone.utc),
        params={
            "universe": universe_name,
            "top_n": top_n,
            "issuer_cap": issuer_cap or 0.0,
            "ad_hoc": ad_hoc,
            "use_float": use_float,
        },
        universe_size=len(tks),
        eligible_size=int(len(eligible)),
        float_coverage=float_coverage,
        portfolio=portfolio,
        scores=scores,
        z_winsorized=scores_e["z_winsorized"].dropna(),
    )
