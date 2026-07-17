"""Next-rebalance preview: diff the live portfolio against a projected one.

The live snapshot on any day inside a month anchors the price lookback to the
last *settled* month-end, so it doesn't move until the month turns over. A
*preview* snapshot (``run_snapshot(..., preview=True)``) instead anchors to the
end of the current month. Diffing the two answers "what changes at the next
rebalance" — entries, exits, and weight/score drift.

Crucially the projected momentum *ranking* is EXACT, not a guess: the 6m/12m
returns read P_{T-1}, P_{T-7}, P_{T-13}, all already-settled prior month-ends
(the current month's own close never enters — it's the unused T-0 anchor). So
membership and scores are fully determined today. The only pieces that drift
intramonth are the *weights*, via same-day market cap and 3y weekly volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from msci_momentum.pipeline import Snapshot


@dataclass
class PortfolioDiff:
    as_of: pd.Timestamp            # the live date the preview was run from
    target_month_end: pd.Timestamp  # month-end the preview anchors on
    entries: pd.DataFrame          # in preview top-N, not in current
    exits: pd.DataFrame            # in current top-N, not in preview
    moves: pd.DataFrame            # in both; weight/score/rank deltas
    current_size: int
    preview_size: int

    @property
    def turnover(self) -> float:
        """One-way turnover: half the sum of absolute weight changes."""
        entered = self.entries["weight"].sum() if not self.entries.empty else 0.0
        exited = self.exits["weight"].sum() if not self.exits.empty else 0.0
        moved = self.moves["delta_weight"].abs().sum() if not self.moves.empty else 0.0
        return 0.5 * (entered + exited + moved)


def _ranked(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Portfolio with a 1-based rank column (by weight, as built)."""
    out = portfolio.copy()
    out["rank"] = range(1, len(out) + 1)
    return out


def diff_snapshots(current: Snapshot, preview: Snapshot) -> PortfolioDiff:
    """Compare a live snapshot to its next-rebalance preview.

    ``current`` and ``preview`` must share a universe and top-N. Pass the
    ``preview=True`` snapshot as the second argument.
    """
    cur = _ranked(current.portfolio)
    prev = _ranked(preview.portfolio)

    cur_ids = cur.index
    prev_ids = prev.index

    entered = prev_ids.difference(cur_ids)
    exited = cur_ids.difference(prev_ids)
    both = cur_ids.intersection(prev_ids)

    cols = ["momentum_score", "weight", "rank", "sector"]
    entries = prev.loc[entered, [c for c in cols if c in prev.columns]].sort_values(
        "weight", ascending=False
    )
    exits = cur.loc[exited, [c for c in cols if c in cur.columns]].sort_values(
        "weight", ascending=False
    )

    moves = pd.DataFrame(
        {
            "weight_now": cur.loc[both, "weight"],
            "weight_next": prev.loc[both, "weight"],
            "delta_weight": prev.loc[both, "weight"] - cur.loc[both, "weight"],
            "score_now": cur.loc[both, "momentum_score"],
            "score_next": prev.loc[both, "momentum_score"],
            "delta_score": prev.loc[both, "momentum_score"] - cur.loc[both, "momentum_score"],
            "rank_now": cur.loc[both, "rank"],
            "rank_next": prev.loc[both, "rank"],
            "delta_rank": prev.loc[both, "rank"] - cur.loc[both, "rank"],
        }
    ).sort_values("delta_weight", key=lambda s: s.abs(), ascending=False)

    return PortfolioDiff(
        as_of=current.date,
        target_month_end=preview.anchor_date,
        entries=entries,
        exits=exits,
        moves=moves,
        current_size=len(cur),
        preview_size=len(prev),
    )
