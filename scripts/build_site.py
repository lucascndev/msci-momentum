"""Build the static GitHub Pages site for the daily Momentum snapshot.

Outputs into ``docs/``:

  docs/index.html              — single-page dashboard (Plotly + table)
  docs/data/YYYY-MM-DD.json    — that day's portfolio + score summary
  docs/data/index.json         — list of available dates (newest first)

Run locally (dry run for today):
    python scripts/build_site.py

Run for a specific date:
    python scripts/build_site.py --date 2026-04-27

The script is idempotent: re-running on the same date overwrites the day's JSON
and regenerates the HTML.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template

from msci_momentum.data import fetch_market_caps, fetch_monthly_closes
from msci_momentum.momentum import build_inputs_for_universe, compute_momentum_scores
from msci_momentum.portfolio import build_portfolio
from msci_momentum.universe import load_universe, tickers as universe_tickers

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data"


PAGE_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>MSCI Momentum — S&P 500 daily snapshot</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1200px; margin: 2rem auto; padding: 0 1.25rem;
    line-height: 1.5;
  }
  h1 { margin-bottom: 0.25rem; }
  .sub { color: #666; margin-top: 0; }
  .banner {
    background: #fff8e1; border: 1px solid #f0d067; padding: 0.75rem 1rem;
    border-radius: 6px; margin: 1.25rem 0; font-size: 0.92rem;
  }
  @media (prefers-color-scheme: dark) {
    .banner { background: #3a300d; border-color: #806810; color: #f5e7b0; }
  }
  .metrics { display: flex; gap: 2rem; flex-wrap: wrap; margin: 1rem 0 1.5rem; }
  .metric { }
  .metric .v { font-size: 1.6rem; font-weight: 600; }
  .metric .k { color: #888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }
  .controls { display: flex; gap: 1rem; align-items: center; margin: 1rem 0; flex-wrap: wrap; }
  select, button {
    font: inherit; padding: 0.4rem 0.7rem; border-radius: 4px;
    border: 1px solid #ccc; background: white;
  }
  @media (prefers-color-scheme: dark) {
    select, button { background: #222; color: #eee; border-color: #444; }
  }
  table { border-collapse: collapse; width: 100%; font-size: 0.93rem; }
  th, td { padding: 0.4rem 0.6rem; text-align: right; border-bottom: 1px solid #eee; }
  th:first-child, td:first-child { text-align: left; }
  th { cursor: pointer; user-select: none; background: #fafafa; }
  @media (prefers-color-scheme: dark) {
    th { background: #1a1a1a; } td, th { border-bottom-color: #333; }
  }
  th.sort-asc::after { content: " ▲"; color: #888; }
  th.sort-desc::after { content: " ▼"; color: #888; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }
  @media (max-width: 800px) { .charts { grid-template-columns: 1fr; } }
  .chart { min-height: 360px; }
  footer { margin-top: 3rem; color: #888; font-size: 0.85rem; }
</style>
</head>
<body>

<h1>MSCI Momentum — S&amp;P 500 snapshot</h1>
<p class="sub">Replication of MSCI Momentum methodology (July 2025). Universe: S&amp;P 500, used as a proxy for MSCI USA.</p>

<div class="banner">
  <strong>Caveat:</strong> uses full market cap, not free-float. Names with large
  strategic holdings (founders, governments) will have inflated parent weight
  versus the official MSCI index. See README for the full methodology coverage table.
</div>

<div class="controls">
  <label>Snapshot:
    <select id="date-picker"></select>
  </label>
  <button id="download-csv">Download portfolio CSV</button>
  <span id="generated-at" style="color:#888;font-size:0.85rem"></span>
</div>

<div class="metrics" id="metrics"></div>

<div class="charts">
  <div id="weights-chart" class="chart"></div>
  <div id="zscore-chart" class="chart"></div>
</div>

<h2>Portfolio</h2>
<div style="overflow-x:auto;">
  <table id="portfolio-table">
    <thead><tr>
      <th data-key="ticker">Ticker</th>
      <th data-key="momentum_score">Score</th>
      <th data-key="parent_weight">Parent wt</th>
      <th data-key="raw_weight">Raw wt</th>
      <th data-key="weight">Weight</th>
    </tr></thead>
    <tbody></tbody>
  </table>
</div>

<footer>
  Generated {{ generated_at }} · data source: yfinance · methodology: MSCI Momentum Indexes (July 2025).
  This is an educational replication, not investment advice.
</footer>

<script>
const DATES = {{ dates_json }};
const FALLBACK_DATE = {{ default_date_json }};

function fmtPct(x) { return (x * 100).toFixed(3) + '%'; }
function fmtNum(x, d=4) { return Number(x).toFixed(d); }

async function loadSnapshot(date) {
  const res = await fetch(`data/${date}.json`, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`Failed to load ${date}`);
  return res.json();
}

function renderMetrics(snap) {
  const el = document.getElementById('metrics');
  const sumW = snap.portfolio.reduce((s, r) => s + r.weight, 0);
  el.innerHTML = `
    <div class="metric"><div class="k">Universe</div><div class="v">${snap.universe_size}</div></div>
    <div class="metric"><div class="k">Eligible</div><div class="v">${snap.eligible_size}</div></div>
    <div class="metric"><div class="k">Selected</div><div class="v">${snap.portfolio.length}</div></div>
    <div class="metric"><div class="k">Σ weight</div><div class="v">${fmtPct(sumW)}</div></div>
    <div class="metric"><div class="k">Top-N target</div><div class="v">${snap.params.top_n}</div></div>
    <div class="metric"><div class="k">Issuer cap</div><div class="v">${fmtPct(snap.params.issuer_cap)}</div></div>
  `;
  document.getElementById('generated-at').textContent = `Generated ${snap.generated_at}`;
}

function renderWeights(snap) {
  const top = snap.portfolio.slice(0, 25);
  Plotly.newPlot('weights-chart', [{
    type: 'bar',
    x: top.map(r => r.ticker),
    y: top.map(r => r.weight * 100),
    marker: { color: '#4a90e2' },
    hovertemplate: '%{x}<br>weight: %{y:.3f}%<extra></extra>',
  }], {
    title: 'Top-25 portfolio weights (%)',
    margin: { t: 40, l: 50, r: 10, b: 80 },
    yaxis: { title: 'weight (%)' },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  }, { displayModeBar: false, responsive: true });
}

function renderZHist(snap) {
  Plotly.newPlot('zscore-chart', [{
    type: 'histogram',
    x: snap.z_winsorized,
    nbinsx: 30,
    marker: { color: '#7b9e89' },
    hovertemplate: 'z ∈ %{x}<br>count: %{y}<extra></extra>',
  }], {
    title: 'Cross-sectional z-score (winsorized at ±3)',
    margin: { t: 40, l: 50, r: 10, b: 50 },
    xaxis: { title: 'z' }, yaxis: { title: 'count' },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  }, { displayModeBar: false, responsive: true });
}

let currentSnap = null;
let sortState = { key: 'weight', dir: 'desc' };

function renderTable() {
  if (!currentSnap) return;
  const rows = [...currentSnap.portfolio];
  const { key, dir } = sortState;
  rows.sort((a, b) => {
    const va = a[key], vb = b[key];
    if (typeof va === 'string') return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
    return dir === 'asc' ? va - vb : vb - va;
  });
  const tbody = document.querySelector('#portfolio-table tbody');
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.ticker}</td>
      <td>${fmtNum(r.momentum_score)}</td>
      <td>${fmtPct(r.parent_weight)}</td>
      <td>${fmtPct(r.raw_weight)}</td>
      <td>${fmtPct(r.weight)}</td>
    </tr>
  `).join('');
  document.querySelectorAll('#portfolio-table th').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.key === key) th.classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');
  });
}

document.querySelectorAll('#portfolio-table th').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    if (sortState.key === key) {
      sortState.dir = sortState.dir === 'asc' ? 'desc' : 'asc';
    } else {
      sortState = { key, dir: key === 'ticker' ? 'asc' : 'desc' };
    }
    renderTable();
  });
});

document.getElementById('download-csv').addEventListener('click', () => {
  if (!currentSnap) return;
  const rows = currentSnap.portfolio;
  const header = ['ticker', 'momentum_score', 'parent_weight', 'raw_weight', 'weight'];
  const csv = [header.join(',')]
    .concat(rows.map(r => header.map(k => r[k]).join(',')))
    .join('\\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `momentum_${currentSnap.date}.csv`;
  a.click(); URL.revokeObjectURL(url);
});

async function show(date) {
  try {
    currentSnap = await loadSnapshot(date);
    renderMetrics(currentSnap);
    renderWeights(currentSnap);
    renderZHist(currentSnap);
    renderTable();
  } catch (e) {
    document.getElementById('metrics').innerHTML =
      `<div class="metric"><div class="k">Error</div><div class="v">${e.message}</div></div>`;
  }
}

(function init() {
  const sel = document.getElementById('date-picker');
  DATES.forEach(d => {
    const o = document.createElement('option');
    o.value = d; o.textContent = d; sel.appendChild(o);
  });
  sel.value = FALLBACK_DATE;
  sel.addEventListener('change', () => show(sel.value));
  show(sel.value);
})();
</script>
</body>
</html>
"""
)


def run_pipeline(rebalance: pd.Timestamp, top_n: int, issuer_cap: float) -> dict:
    members = load_universe("sp500")
    tks = universe_tickers(members)

    inputs = build_inputs_for_universe(tks, rebalance, country="USA", use_cache=True)
    scores = compute_momentum_scores(inputs, use_only_6m=False)

    monthly = fetch_monthly_closes(tks, rebalance, use_cache=True)
    mcap = fetch_market_caps(tks, rebalance, monthly, use_cache=True)

    eligible = scores["combined"].dropna().index.intersection(mcap.index)
    scores_e = scores.loc[eligible]
    mcap_e = mcap.loc[eligible]

    portfolio = build_portfolio(
        scores_e, parent_mcap=mcap_e, n=top_n, issuer_cap=issuer_cap or None
    )

    portfolio_records = [
        {
            "ticker": str(idx),
            "momentum_score": float(row["momentum_score"]),
            "parent_weight": float(row["parent_weight"]),
            "raw_weight": float(row["raw_weight"]),
            "weight": float(row["weight"]),
        }
        for idx, row in portfolio.iterrows()
    ]
    z_winsorized = (
        scores["z_winsorized"].dropna().astype(float).round(4).tolist()
    )
    return {
        "date": rebalance.strftime("%Y-%m-%d"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {"top_n": top_n, "issuer_cap": issuer_cap, "universe": "sp500"},
        "universe_size": len(tks),
        "eligible_size": int(len(eligible)),
        "portfolio": portfolio_records,
        "z_winsorized": z_winsorized,
    }


def write_snapshot(snapshot: dict) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{snapshot['date']}.json"
    path.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
    return path


def write_index() -> list[str]:
    DATA.mkdir(parents=True, exist_ok=True)
    dates = sorted(
        (p.stem for p in DATA.glob("*.json") if p.stem != "index"),
        reverse=True,
    )
    (DATA / "index.json").write_text(json.dumps(dates), encoding="utf-8")
    return dates


def write_html(dates: list[str], default_date: str) -> Path:
    html = PAGE_TEMPLATE.render(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dates_json=json.dumps(dates),
        default_date_json=json.dumps(default_date),
    )
    path = DOCS / "index.html"
    DOCS.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    # Empty .nojekyll so GitHub Pages serves files starting with underscore if any
    (DOCS / ".nojekyll").touch()
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD; defaults to today (UTC).")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--issuer-cap", type=float, default=0.05)
    args = p.parse_args(argv)

    rebalance = (
        pd.Timestamp(args.date)
        if args.date
        else pd.Timestamp(datetime.now(timezone.utc).date())
    )
    snapshot = run_pipeline(rebalance, args.top_n, args.issuer_cap)
    snap_path = write_snapshot(snapshot)
    dates = write_index()
    html_path = write_html(dates, snapshot["date"])

    print(f"Wrote {snap_path.relative_to(ROOT)}")
    print(f"Wrote {html_path.relative_to(ROOT)} ({len(dates)} snapshots indexed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
