"""CLI for computing MSCI Momentum portfolios."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from msci_momentum.pipeline import run_snapshot
from msci_momentum.preview import diff_snapshots


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="momentum",
        description="MSCI Momentum index replication. Outputs the portfolio "
        "weight a stock would receive on a given rebalance date.",
    )
    p.add_argument(
        "--universe",
        default="sp500",
        help="Universe to use (default: sp500). The S&P 500 stands in for "
        "MSCI USA / MSCI World — momentum is RELATIVE so the universe matters.",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Rebalance date (YYYY-MM-DD). Defaults to today.",
    )
    p.add_argument(
        "--ticker",
        default=None,
        help="If set, only print this ticker's row (case-insensitive).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of constituents to select (default: 100, ~20%% of S&P 500). "
        "MSCI USA Momentum uses ~125; tune for your universe.",
    )
    p.add_argument(
        "--issuer-cap",
        type=float,
        default=0.05,
        help="Per-issuer weight cap (default 0.05 = 5%%). Set to 0 to disable.",
    )
    p.add_argument(
        "--ad-hoc",
        action="store_true",
        help="Use only 6m momentum (Appendix III ad-hoc rebalance mode).",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Diff the live portfolio against the projected NEXT rebalance "
        "(anchored on the upcoming month-end). Membership and scores are EXACT "
        "(they read already-settled month-ends); only weights drift intramonth. "
        "Shows entries/exits/weight drift.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk price/share cache.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Rows to print when no --ticker is given (default 20).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _fmt_pct(x: float) -> str:
    return f"{x:+.4%}" if x else "  0.0000%"


def _print_membership(d, top_n: int) -> None:
    if d.entries.empty and d.exits.empty:
        print("  No membership changes — same top-N.")
        return
    print(f"  ENTRIES ({len(d.entries)}) — joining the top-{top_n}")
    if d.entries.empty:
        print("    (none)")
    else:
        e = d.entries.copy()
        e["weight"] = e["weight"].map(lambda x: f"{x:.4%}")
        e["momentum_score"] = e["momentum_score"].map(lambda x: f"{x:.4f}")
        print(e.to_string().replace("\n", "\n  "))
    print(f"  EXITS ({len(d.exits)}) — dropping out")
    if d.exits.empty:
        print("    (none)")
    else:
        x = d.exits.copy()
        x["weight"] = x["weight"].map(lambda v: f"{v:.4%}")
        x["momentum_score"] = x["momentum_score"].map(lambda v: f"{v:.4f}")
        print(x.to_string().replace("\n", "\n  "))


def _print_moves(d, limit: int) -> None:
    moves = d.moves.head(limit).copy()
    if moves.empty:
        print("    (none)")
        return
    show = pd.DataFrame(
        {
            "weight_now": moves["weight_now"].map(lambda v: f"{v:.4%}"),
            "weight_next": moves["weight_next"].map(lambda v: f"{v:.4%}"),
            "delta_weight": moves["delta_weight"].map(_fmt_pct),
            "delta_score": moves["delta_score"].map(lambda v: f"{v:+.4f}"),
            "delta_rank": moves["delta_rank"].map(lambda v: f"{int(v):+d}"),
        }
    )
    print(show.to_string().replace("\n", "\n  "))


def _run_preview(args: argparse.Namespace, rebalance: pd.Timestamp) -> int:
    """Run the live snapshot + both projection horizons and print their diffs.

    Horizon 1 (this month-end) is EXACT — it reads only settled month-ends.
    Horizon 2 (next month-end) is PROVISIONAL — it depends on the in-progress
    month's close and firms up as the month proceeds.
    """
    common = dict(
        universe_name=args.universe,
        top_n=args.top_n,
        issuer_cap=args.issuer_cap or None,
        ad_hoc=args.ad_hoc,
        use_cache=not args.no_cache,
    )
    current = run_snapshot(rebalance, preview=False, **common)
    h1 = run_snapshot(rebalance, preview=True, preview_months_ahead=1, **common)
    h2 = run_snapshot(rebalance, preview=True, preview_months_ahead=2, **common)
    d1 = diff_snapshots(current, h1)   # live -> next rebalance
    d2 = diff_snapshots(h1, h2)        # next rebalance -> the one after

    print(
        f"# NEXT-REBALANCE PREVIEW  universe={args.universe}  top_n={args.top_n}  "
        f"live={rebalance.date()}",
        file=sys.stderr,
    )

    print(
        f"\n=== HORIZON 1: next rebalance @ {d1.target_month_end.date()} "
        f"(EXACT — settled month-ends; est. turnover {d1.turnover:.2%}) ==="
    )
    _print_membership(d1, args.top_n)
    print(f"\n  Largest weight moves (top {args.limit}):")
    _print_moves(d1, args.limit)

    print(
        f"\n=== HORIZON 2: following rebalance @ {d2.target_month_end.date()} "
        f"(PROVISIONAL — depends on the in-progress month; will change) ==="
    )
    print("  Change vs. horizon 1:")
    _print_membership(d2, args.top_n)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    rebalance = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()

    if args.preview:
        return _run_preview(args, rebalance)

    snap = run_snapshot(
        rebalance,
        universe_name=args.universe,
        top_n=args.top_n,
        issuer_cap=args.issuer_cap or None,
        ad_hoc=args.ad_hoc,
        use_cache=not args.no_cache,
    )
    portfolio = snap.portfolio
    scores = snap.scores

    print(
        f"# Universe: {args.universe} ({snap.universe_size} tickers)  "
        f"Rebalance: {rebalance.date()}  Top-N: {args.top_n}  "
        f"Cap: {args.issuer_cap:.0%}  "
        f"Float coverage: {snap.float_coverage}/{snap.universe_size}",
        file=sys.stderr,
    )

    if args.ticker:
        tk = args.ticker.upper().replace(".", "-")
        if tk in portfolio.index:
            row = portfolio.loc[tk]
            print(f"{tk}")
            print(f"  momentum_score : {row['momentum_score']:.4f}")
            print(f"  parent_weight  : {row['parent_weight']:.4%}")
            print(f"  weight         : {row['weight']:.4%}")
        elif tk in scores.index:
            row = scores.loc[tk]
            print(f"{tk} (NOT SELECTED into top-{args.top_n})")
            print(f"  momentum_score : {row['momentum_score']:.4f}")
            print(f"  z (winsorized) : {row['z_winsorized']:.4f}")
            print(f"  z (raw)        : {row['z_unwinsorized']:.4f}")
            print(f"  weight         : 0.0000%")
        else:
            print(f"{tk} not in universe or missing momentum value", file=sys.stderr)
            return 1
        return 0

    head = portfolio.head(args.limit)
    out = head.copy()
    out["weight"] = out["weight"].map(lambda x: f"{x:.4%}")
    out["parent_weight"] = out["parent_weight"].map(lambda x: f"{x:.4%}")
    out["raw_weight"] = out["raw_weight"].map(lambda x: f"{x:.4%}")
    out["momentum_score"] = out["momentum_score"].map(lambda x: f"{x:.4f}")
    print(out.to_string())
    print(
        f"\n# Selected {len(portfolio)} of {args.top_n} target  "
        f"sum(weight)={portfolio['weight'].sum():.4%}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
