"""CLI for computing MSCI Momentum portfolios."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from msci_momentum.pipeline import run_snapshot


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    rebalance = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()

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
