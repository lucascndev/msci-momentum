"""Streamlit GUI for the MSCI Momentum replication.

Launch with either:
    streamlit run src/msci_momentum/app.py
    momentum-gui                              # console-script wrapper
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from msci_momentum.pipeline import run_snapshot


@st.cache_data(show_spinner=False)
def _run_pipeline(
    universe_name: str,
    rebalance_iso: str,
    top_n: int,
    issuer_cap: float,
    ad_hoc: bool,
    use_cache: bool,
) -> dict:
    snap = run_snapshot(
        pd.Timestamp(rebalance_iso),
        universe_name=universe_name,
        top_n=top_n,
        issuer_cap=issuer_cap or None,
        ad_hoc=ad_hoc,
        use_cache=use_cache,
    )
    return {
        "universe_size": snap.universe_size,
        "eligible_size": snap.eligible_size,
        "float_coverage": snap.float_coverage,
        "scores": snap.scores,
        "portfolio": snap.portfolio,
    }


def _render():
    st.set_page_config(page_title="MSCI Momentum", layout="wide")
    st.title("MSCI Momentum — replication")
    st.caption(
        "Point-in-time replication of MSCI Momentum methodology (July 2025). "
        "Universe is currently S&P 500 (proxy for MSCI USA)."
    )

    with st.sidebar:
        st.header("Parameters")
        universe_name = st.selectbox("Universe", ["sp500"], index=0)
        rebalance_date = st.date_input(
            "Rebalance date", value=pd.Timestamp.today().normalize().date()
        )
        top_n = st.slider("Top-N", min_value=10, max_value=300, value=100, step=10)
        issuer_cap_pct = st.slider(
            "Issuer cap (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5,
            help="Per-issuer weight cap. 0 disables.",
        )
        ad_hoc = st.checkbox(
            "Ad-hoc rebalance (6m only, Appendix III)", value=False
        )
        use_cache = not st.checkbox("Bypass on-disk cache", value=False)
        ticker_filter = st.text_input(
            "Single-ticker view (optional)", value=""
        ).strip().upper().replace(".", "-")
        run = st.button("Run", type="primary")

    if not run:
        st.info("Set parameters in the sidebar and click **Run**.")
        return

    with st.spinner("Fetching prices and computing scores…"):
        try:
            result = _run_pipeline(
                universe_name=universe_name,
                rebalance_iso=pd.Timestamp(rebalance_date).strftime("%Y-%m-%d"),
                top_n=top_n,
                issuer_cap=issuer_cap_pct / 100.0,
                ad_hoc=ad_hoc,
                use_cache=use_cache,
            )
        except Exception as exc:  # noqa: BLE001 — surface to UI
            st.error(f"Pipeline failed: {exc}")
            st.exception(exc)
            return

    portfolio: pd.DataFrame = result["portfolio"]
    scores: pd.DataFrame = result["scores"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe", result["universe_size"])
    c2.metric("Eligible", result["eligible_size"])
    c3.metric("Float coverage", result["float_coverage"])
    c4.metric("Selected", len(portfolio))
    c5.metric("Σ weight", f"{portfolio['weight'].sum():.2%}")

    if ticker_filter:
        st.subheader(f"Ticker: {ticker_filter}")
        if ticker_filter in portfolio.index:
            row = portfolio.loc[ticker_filter]
            st.write(
                {
                    "momentum_score": float(row["momentum_score"]),
                    "parent_weight": float(row["parent_weight"]),
                    "weight": float(row["weight"]),
                    "selected": True,
                }
            )
        elif ticker_filter in scores.index:
            row = scores.loc[ticker_filter]
            st.warning(f"{ticker_filter} not selected into top-{top_n}.")
            st.write(
                {
                    "momentum_score": float(row["momentum_score"]),
                    "z_winsorized": float(row["z_winsorized"]),
                    "z_unwinsorized": float(row["z_unwinsorized"]),
                    "weight": 0.0,
                }
            )
        else:
            st.error(f"{ticker_filter} not in universe or missing momentum value.")

    tab_port, tab_dist, tab_raw = st.tabs(
        ["Portfolio", "Score distribution", "Raw scores"]
    )

    with tab_port:
        st.subheader("Portfolio weights")
        display = portfolio.copy()
        display["weight"] = display["weight"].astype(float)
        display["parent_weight"] = display["parent_weight"].astype(float)
        st.dataframe(
            display.style.format(
                {
                    "momentum_score": "{:.4f}",
                    "raw_weight": "{:.4%}",
                    "parent_weight": "{:.4%}",
                    "weight": "{:.4%}",
                }
            ),
            use_container_width=True,
            height=520,
        )
        st.bar_chart(portfolio["weight"].head(25))
        st.download_button(
            "Download portfolio CSV",
            data=portfolio.to_csv().encode("utf-8"),
            file_name=f"momentum_portfolio_{rebalance_date}_top{top_n}.csv",
            mime="text/csv",
        )

    with tab_dist:
        st.subheader("Cross-sectional z-score distribution")
        z = scores["z_winsorized"].dropna()
        st.write(
            {
                "count": int(z.size),
                "mean": float(z.mean()),
                "stdev": float(z.std(ddof=1)),
                "winsorized_at_+3": int((z >= 3.0).sum()),
                "winsorized_at_-3": int((z <= -3.0).sum()),
            }
        )
        hist = (
            pd.cut(z, bins=30)
            .value_counts()
            .sort_index()
            .rename_axis("bin")
            .reset_index(name="count")
        )
        hist["bin"] = hist["bin"].astype(str)
        st.bar_chart(hist.set_index("bin"))

    with tab_raw:
        st.subheader("All scored names")
        st.dataframe(scores.sort_values("momentum_score", ascending=False), height=520)
        st.download_button(
            "Download scores CSV",
            data=scores.to_csv().encode("utf-8"),
            file_name=f"momentum_scores_{rebalance_date}.csv",
            mime="text/csv",
        )


def run() -> None:
    """Console-script entry point: re-exec via `streamlit run` on this file."""
    import subprocess

    app_path = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


def _in_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001
        return False


if _in_streamlit_runtime():
    _render()
