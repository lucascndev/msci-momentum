# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A point-in-time replication of the **MSCI Momentum Indexes** methodology
(July 2025 PDF, MSCI Inc.). Given an equity universe and a rebalance date,
it computes each stock's Momentum Score and the weight it would receive in a
Momentum-tilted portfolio that mirrors how MSCI builds e.g. MSCI USA
Momentum or MSCI World Momentum.

## Commands

```bash
pip install -e .[dev]               # install with dev deps (pytest)
pytest                              # run unit tests (no network)
pytest tests/test_momentum.py::test_score_map_positive_branch  # single test
momentum --date 2026-04-15 --top-n 100               # full top-100 portfolio
momentum --date 2026-04-15 --ticker AAPL             # single stock's weight
momentum --date 2026-04-15 --ad-hoc                  # 6m-only (Appendix III)
momentum --no-cache ...                              # bypass on-disk cache
```

`MOMENTUM_CACHE` env var overrides the cache directory
(default `~/.cache/msci-momentum`).

## Architecture

The pipeline is **strictly cross-sectional**: a single stock has no Momentum
Score in isolation. The math at every step requires the universe's mean and
stdev. This shapes the module boundaries:

```
universe.py → list of UniverseMember
    ↓
data.py     → monthly closes, weekly closes, market caps (yfinance)
risk_free.py → annualized short-term rate at T (yfinance ^IRX for USD)
    ↓
momentum.py → MomentumInputs → DataFrame of per-ticker raw, riskadj, z, score
    ↓
portfolio.py → top-N selection, weighting, iterative 5% issuer cap
    ↓
cli.py      → orchestration + pretty-print
```

Key design points a future change will probably trip over:

* **Cache keys include a hash of the ticker list** (`_ticker_hash` in
  `data.py`). This is because the cache is shared across universes — if you
  cache `monthly_2026-04-30.csv` after fetching for 30 tickers, then call
  again with 500 tickers on the same date, the cache returns 30 columns and
  silently truncates the universe. The hash prevents that.

* **MSCI Momentum is a *combined* z-score**: per-stock risk-adjusted 6m and
  12m returns are *each* z-scored across the universe, then averaged 50/50,
  then z-scored *again*, then winsorized at ±3, then mapped to a positive
  score via `1+Z if Z>0 else 1/(1−Z)`. Skipping any of these steps changes
  the ranking and weights.

* **Issuer cap is iterative.** Capping a 30% weight at 5% spills 25% across
  the rest, which can push *another* name over the cap. The loop in
  `apply_issuer_cap` keeps redistributing until stable. If `cap × N < 1`
  the cap is mathematically infeasible — we raise rather than oscillate.

* **Market cap uses *full* mcap, not free-float.** MSCI uses free-float;
  yfinance doesn't expose it cleanly. This will overstate the parent weight
  of names with large strategic holdings (founders, governments). A future
  improvement is to plug in a float factor source.

* **Risk-free rate is single-country.** `risk_free.RATE_TICKERS` only has
  USD (`^IRX`). MSCI Methodology Appendix IV lists the per-country tickers
  for ~50 markets — needed before this works for MSCI World / EM. The 6m
  and 12m horizon rates are computed as `annual × months/12` (simple
  pro-rate, not compounded).

## What is intentionally NOT implemented

If a user asks for these, they need to be added — they don't already exist:

* Quarterly rebalancing schedule and turnover buffer (sections 3.1.1, 3.1.2)
* The dynamic Top-N algorithm in Appendix I (currently `--top-n` is manual)
* Volatility-trigger detection for Appendix III (the `--ad-hoc` flag just
  switches to 6m-only; it doesn't *decide* whether to ad-hoc rebalance)
* Corporate event handling between rebalances (section 3.2)
* The MSCI Momentum *Tilt* index variant (Appendix V)
* Universes other than S&P 500
* Free-float adjustment of market cap

The mapping from MSCI methodology section → implementation status is in
`README.md` (the "Methodology coverage" table); update both when adding a
section.

## Testing

`tests/test_momentum.py` covers the math with synthetic data — no network
required, runs in <1s. The CLI smoke run hits yfinance and is not in CI;
run it manually after touching `data.py` or `cli.py`.
