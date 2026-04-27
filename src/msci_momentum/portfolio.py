"""Portfolio construction: selection, weighting, and 5% issuer cap.

MSCI Momentum index construction (sections 2.3, 2.4 + Appendix II):

  1. Rank parent universe by *unwinsorized* combined Z (descending). Pick top N.
  2. Weight = MomentumScore * parent_mcap_weight, then normalize to 100%.
  3. Apply issuer-weight cap (5% on broad indexes; max parent mcap weight on
     narrow ones). We use the broad-index cap by default — S&P 500 fits.

The cap is enforced iteratively: any name above the cap is set to the cap, and
the excess is redistributed proportionally to the uncapped names. We repeat
until no name exceeds the cap or all names are capped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_top_n(scores: pd.DataFrame, n: int, parent_mcap: pd.Series) -> pd.Index:
    """Top-N tickers by unwinsorized Z; ties broken by parent mcap weight.

    Implements section 2.3 — including the tie-break rule that "the security
    having a higher weight in the Parent Index is given a higher rank".
    """
    z = scores["z_unwinsorized"].dropna()
    if z.empty:
        return pd.Index([])

    # Align tie-break key to z's index, fill missing with 0 (ranks them last).
    tiebreak = parent_mcap.reindex(z.index).fillna(0.0)
    ranked = pd.DataFrame({"z": z, "tb": tiebreak}).sort_values(
        ["z", "tb"], ascending=[False, False]
    )
    return ranked.head(n).index


def momentum_weights(
    scores: pd.DataFrame,
    parent_mcap: pd.Series,
    selected: pd.Index | None = None,
) -> pd.Series:
    """Score * parent_mcap_weight, restricted to ``selected``, normalized."""
    score = scores["momentum_score"]
    if selected is not None:
        score = score.loc[score.index.intersection(selected)]

    parent = parent_mcap.reindex(score.index).dropna()
    score = score.loc[parent.index].dropna()
    parent = parent.loc[score.index]

    parent_w = parent / parent.sum()
    raw = score * parent_w
    if raw.sum() == 0:
        return raw
    return raw / raw.sum()


def apply_issuer_cap(
    weights: pd.Series, cap: float = 0.05, max_iter: int = 100
) -> pd.Series:
    """Iteratively cap any single weight at ``cap`` and redistribute excess.

    For broad regional/country indexes MSCI uses a 5% issuer cap (Appendix II).
    We treat each ticker as its own issuer — fine for v1; multi-class issuers
    would need a separate issuer mapping.
    """
    if weights.empty:
        return weights
    w = weights.astype(float).copy()
    if w.sum() <= 0:
        return w
    w = w / w.sum()

    if cap * len(w) < 1.0 - 1e-12:
        # Infeasible: even if every name were at the cap, total < 100%.
        # MSCI's cap algorithm assumes enough names to absorb capped excess;
        # we surface the issue rather than oscillate.
        raise ValueError(
            f"Issuer cap {cap:.2%} infeasible for {len(w)} names "
            f"(max coverage {cap * len(w):.1%}); raise the cap or grow the universe"
        )

    for _ in range(max_iter):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w.loc[over] = cap
        room = w[~over]
        if room.sum() <= 0:
            # Everyone is at the cap — proportional redistribution impossible.
            break
        w.loc[~over] = room + excess * (room / room.sum())

    # Guard against drift from the loop.
    s = w.sum()
    if s > 0:
        w = w / s
    return w


def build_portfolio(
    scores: pd.DataFrame,
    parent_mcap: pd.Series,
    n: int,
    issuer_cap: float | None = 0.05,
) -> pd.DataFrame:
    """End-to-end: select top-N, weight, optionally cap.

    Returns a DataFrame indexed by ticker with columns:
      momentum_score, parent_weight, raw_weight, weight
    """
    selected = select_top_n(scores, n=n, parent_mcap=parent_mcap)
    sub_scores = scores.loc[selected.intersection(scores.index)]
    sub_parent = parent_mcap.reindex(sub_scores.index).dropna()
    sub_scores = sub_scores.loc[sub_parent.index]

    parent_w = sub_parent / sub_parent.sum()
    raw = sub_scores["momentum_score"] * parent_w
    raw_norm = raw / raw.sum() if raw.sum() else raw
    final = apply_issuer_cap(raw_norm, cap=issuer_cap) if issuer_cap else raw_norm

    return pd.DataFrame(
        {
            "momentum_score": sub_scores["momentum_score"],
            "parent_weight": parent_w,
            "raw_weight": raw_norm,
            "weight": final,
        }
    ).sort_values("weight", ascending=False)
