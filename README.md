# msci-momentum

Replicates the [MSCI Momentum Indexes methodology](https://www.msci.com/index/methodology/latest/MOM)
(July 2025) against a configurable equity universe. Given a universe and a
rebalance date, it computes each stock's Momentum Score and the weight it
would receive in a Momentum-tilted portfolio.

## Why a universe matters

Momentum is **relative**. The pipeline z-scores risk-adjusted returns across
the cross-section, so a single stock's score depends on every other stock it's
competing against. "What weight should AAPL have?" only has an answer once you
fix the universe (S&P 500, MSCI World, a custom basket, ...).

v1 ships with **S&P 500** as a stand-in for MSCI USA / MSCI World. Adding
universes means providing the constituent list and per-country risk-free rates
(Appendix IV of the MSCI document).

## Setup

Requires Python 3.10+.

```bash
# 1. clone & enter the project
git clone https://github.com/lucascndev/msci-momentum.git
cd msci-momentum

# 2. create a virtualenv and activate it
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (Git Bash):    source .venv/Scripts/activate
# macOS / Linux:         source .venv/bin/activate

# 3. install the package + dev tools (pytest)
pip install -e .[dev]
```

This installs `yfinance`, `pandas`, `numpy`, `lxml` and registers a `momentum`
command in the active venv.

Verify the install:

```bash
pytest                  # 16 unit tests, ~1s, no network
momentum --help
```

### Windows notes

- **`pip` not on PATH globally**: don't worry about it — once the venv is
  activated, `pip` resolves to `.venv\Scripts\pip.exe`. If activation isn't
  working for you, you can always invoke the venv binaries directly:
  `.venv\Scripts\pip.exe install -e .[dev]`,
  `.venv\Scripts\momentum.exe --help`,
  `.venv\Scripts\pytest.exe`.
- **`Unable to copy '...\venvlauncher.exe'`** when running `python -m venv .venv`:
  this happens with some non-standard Python installs (e.g. nuget / embeddable
  distributions). It's safe to ignore as long as `.venv\Scripts\` ends up
  containing `python.exe` and `pip.exe` (it does in the cases I've seen).
- **PowerShell execution policy**: if `.\.venv\Scripts\Activate.ps1` errors
  with "running scripts is disabled", run
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once,
  or use `cmd.exe` (`.venv\Scripts\activate.bat`) or Git Bash instead.

## Running it

The CLI fetches prices and market caps from Yahoo Finance, runs the MSCI
pipeline, and prints either a portfolio table or a single ticker's weight.

### What to expect on the first run

The first call for a given date downloads:

- 18 months of monthly closes (one batched call, ~5s)
- 3 years of weekly closes (one batched call, ~30s)
- `sharesOutstanding` for every S&P 500 ticker — **one HTTP call per ticker**,
  paced at 50 ms each. This takes **~5 minutes** for the full 503 tickers.

Everything is cached under `~/.cache/msci-momentum/` (override with the
`MOMENTUM_CACHE` env var). Subsequent runs for the same date complete in
seconds. Use `--no-cache` to force a refetch.

### Look up one stock's weight

This is the core question the project answers — "what weight should X have
in a Momentum portfolio?":

```bash
momentum --date 2026-04-15 --ticker GOOGL
```

Output:

```
GOOGL
  momentum_score : 2.0680
  parent_weight  : 12.5394%
  weight         : 5.0000%
```

If the stock didn't make the cut you get its score and a `NOT SELECTED`
notice — useful to see *why* it was filtered out:

```bash
momentum --date 2026-04-15 --ticker AAPL
```

```
AAPL (NOT SELECTED into top-100)
  momentum_score : 0.9561
  z (winsorized) : -0.0459
  z (raw)        : -0.0459
  weight         : 0.0000%
```

### See the whole portfolio

Drop `--ticker` to print the top constituents (default: top 20 by weight):

```bash
momentum --date 2026-04-15 --top-n 100 --limit 10
```

```
      momentum_score parent_weight raw_weight   weight
MU            3.6269       3.5032%    5.5295%  5.0000%
GOOGL         2.0680      12.5394%   11.2854%  5.0000%
WMT           1.9998       6.4763%    5.6365%  5.0000%
GOOG          2.0484      11.6416%   10.3781%  5.0000%
XOM           2.7518       3.8708%    4.6356%  5.0000%
JNJ           2.7728       3.4248%    4.1328%  4.9567%
CAT           3.1103       2.4174%    3.2723%  3.9245%
LRCX          3.0274       2.0942%    2.7592%  3.3092%
AMD           1.7350       3.5461%    2.6776%  3.2113%
AMAT          2.9207       2.0701%    2.6313%  3.1559%
```

### All flags

| Flag | Default | What it does |
|------|---------|---|
| `--universe` | `sp500` | Constituent universe. Currently only `sp500` is shipped. |
| `--date` | today | Rebalance date `YYYY-MM-DD`. Picks the latest month-end ≤ this date. |
| `--ticker` | (none) | Print one ticker's row instead of the portfolio. Use yfinance symbols (e.g. `BRK-B`, not `BRK.B`). |
| `--top-n` | `100` | Number of constituents to select. MSCI USA Momentum uses ~125. |
| `--issuer-cap` | `0.05` | Per-issuer weight cap (5%). Pass `0` to disable. |
| `--ad-hoc` | off | Use 6-month momentum only (Appendix III ad-hoc rebalance mode). |
| `--no-cache` | off | Bypass the on-disk cache. |
| `--limit` | `20` | How many portfolio rows to print (ignored with `--ticker`). |
| `-v` | off | Verbose logging. **Don't combine with the full S&P 500 — yfinance debug logs are huge.** |

## Methodology coverage

| MSCI methodology section          | Status |
|-----------------------------------|--------|
| 2.2 Momentum value (6m + 12m)     | ✓      |
| 2.2.1 Risk adjustment by 3y σ     | ✓      |
| 2.2.2 z-score, average, re-z, winsorize ±3, MSCI score map | ✓ |
| 2.3 Top-N selection w/ tie-break  | ✓      |
| 2.4 Score × parent mcap weight    | ✓      |
| Appendix II 5% issuer cap         | ✓ (each ticker = one issuer) |
| Appendix III ad-hoc rebalance     | ✓ (`--ad-hoc` flag; vol trigger NOT computed) |
| Appendix IV per-country rates     | partial — USA only (`^IRX`) |
| Appendix I dynamic N selection    | not implemented (`--top-n` is manual) |
| Quarterly rebalance schedule      | n/a — point-in-time tool |
| Turnover buffer (3.1.2)           | not implemented |
| Buffer rules (3.1.1)              | not implemented |
| Corporate event handling (3.2)    | not implemented |
| Tilt index (Appendix V)           | not implemented |

## Data caveats

* **Prices**: yfinance, monthly + weekly closes. Split-adjusted but **not**
  dividend-adjusted (matches "local price returns" in MSCI terminology).
* **Market cap**: `sharesOutstanding × last_close`. MSCI uses *free-float*
  market cap; full mcap is a practical proxy but will overstate names with
  large strategic holdings (founders, governments, cross-holdings).
* **Risk-free rate**: 13-week T-Bill yield (`^IRX`) at the rebalance date,
  pro-rated to 6m and 12m horizons.

## Tests

```bash
pytest
```

The math is covered by synthetic-data unit tests (no network).

## Project layout

```
src/msci_momentum/
  universe.py    # constituent loader (S&P 500 from Wikipedia, cached)
  data.py        # yfinance price + market-cap fetching
  risk_free.py   # local short-term rates
  momentum.py    # MSCI scoring pipeline
  portfolio.py   # selection, weighting, issuer cap
  cli.py         # `momentum` entry point
tests/
  test_momentum.py
```
