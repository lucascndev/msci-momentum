"""Unit tests for the MSCI Momentum scoring pipeline.

Uses synthetic price series so no network access is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msci_momentum.data import month_end_closes_at_offset
from msci_momentum.momentum import (
    MomentumInputs,
    _msci_score_from_z,
    annualized_weekly_volatility,
    compute_momentum_scores,
)
from msci_momentum.portfolio import apply_issuer_cap, build_portfolio, select_top_n


# ---------- score map ----------


def test_score_map_positive_branch():
    z = pd.Series([0.5, 1.0, 2.0, 3.0])
    out = _msci_score_from_z(z)
    assert list(out) == [1.5, 2.0, 3.0, 4.0]


def test_score_map_negative_branch():
    z = pd.Series([-0.5, -1.0, -2.0, -3.0])
    out = _msci_score_from_z(z)
    np.testing.assert_allclose(out, [1 / 1.5, 0.5, 1 / 3, 0.25])


def test_score_map_zero_goes_negative_branch():
    # Spec: positive when Z > 0, else 1/(1-Z); Z=0 -> 1.0 either way.
    out = _msci_score_from_z(pd.Series([0.0]))
    assert out.iloc[0] == 1.0


# ---------- volatility ----------


def test_annualized_weekly_volatility_constant_returns():
    # Weekly returns of constant 1% -> stdev 0 -> sigma 0 (then NaN'd downstream).
    n = 200
    prices = pd.DataFrame(
        {"X": [100 * (1.01 ** i) for i in range(n)]},
        index=pd.date_range("2020-01-01", periods=n, freq="W"),
    )
    sigma = annualized_weekly_volatility(prices)
    assert sigma["X"] == pytest.approx(0.0, abs=1e-10)


def test_annualized_weekly_volatility_drops_short_history():
    n = 30
    prices = pd.DataFrame(
        {"X": np.linspace(100, 130, n)},
        index=pd.date_range("2024-01-01", periods=n, freq="W"),
    )
    sigma = annualized_weekly_volatility(prices)
    assert np.isnan(sigma["X"])


# ---------- end-to-end momentum scoring ----------


def _toy_inputs() -> MomentumInputs:
    """Three stocks: A is a strong winner, B middling, C a loser."""
    p1 = pd.Series({"A": 130.0, "B": 110.0, "C": 95.0})
    p7 = pd.Series({"A": 100.0, "B": 100.0, "C": 100.0})
    p13 = pd.Series({"A": 90.0, "B": 100.0, "C": 105.0})
    sigma = pd.Series({"A": 0.30, "B": 0.20, "C": 0.25})
    return MomentumInputs(
        p_t_minus_1=p1,
        p_t_minus_7=p7,
        p_t_minus_13=p13,
        sigma_annual=sigma,
        rf_6m=0.025,
        rf_12m=0.05,
    )


def test_compute_momentum_scores_orders_winners_above_losers():
    inputs = _toy_inputs()
    out = compute_momentum_scores(inputs)
    assert out.loc["A", "momentum_score"] > out.loc["B", "momentum_score"]
    assert out.loc["B", "momentum_score"] > out.loc["C", "momentum_score"]


def test_compute_momentum_scores_winsorizes_at_three():
    # Make A a wild outlier.
    inputs = _toy_inputs()
    inputs.p_t_minus_1["A"] = 1000.0
    out = compute_momentum_scores(inputs)
    # Z is winsorized at +/-3, so the max score is 1+3 = 4.0 and min is 0.25.
    assert out["z_winsorized"].max() <= 3.0 + 1e-9
    assert out["z_winsorized"].min() >= -3.0 - 1e-9
    assert out["momentum_score"].max() <= 4.0 + 1e-9


def test_compute_momentum_scores_ad_hoc_uses_only_6m():
    inputs = _toy_inputs()
    out = compute_momentum_scores(inputs, use_only_6m=True)
    # In ad-hoc mode the combined value equals the risk-adjusted 6m value.
    np.testing.assert_allclose(
        out["combined"].values, out["riskadj_6m"].values, rtol=1e-12
    )


def test_combined_falls_back_to_6m_when_12m_missing():
    inputs = _toy_inputs()
    inputs.p_t_minus_13["A"] = np.nan  # 12m unavailable for A
    out = compute_momentum_scores(inputs)
    assert pd.notna(out.loc["A", "combined"])
    np.testing.assert_allclose(
        out.loc["A", "combined"], out.loc["A", "z_6m"], rtol=1e-12
    )


# ---------- portfolio construction ----------


def test_select_top_n_breaks_ties_by_parent_mcap():
    scores = pd.DataFrame(
        {"z_unwinsorized": [1.0, 1.0, 0.5]}, index=["A", "B", "C"]
    )
    parent = pd.Series({"A": 1.0, "B": 5.0, "C": 10.0})
    sel = select_top_n(scores, n=2, parent_mcap=parent)
    # A and B tie on Z; B has bigger parent mcap so wins the tie.
    assert list(sel) == ["B", "A"]


def test_apply_issuer_cap_caps_and_redistributes():
    # Feasible 5% cap: 30 names, two over-weight at 30% / 20%, rest tiny.
    raw = {f"S{i}": 0.5 / 28 for i in range(28)}
    raw["BIG"] = 0.30
    raw["MID"] = 0.20
    w = pd.Series(raw)
    out = apply_issuer_cap(w, cap=0.05)
    assert out["BIG"] == pytest.approx(0.05, abs=1e-9)
    assert out["MID"] == pytest.approx(0.05, abs=1e-9)
    # Excess of 0.40 redistributes to the 28 small names; each grows.
    assert (out.drop(["BIG", "MID"]) > 0.5 / 28).all()
    assert out.sum() == pytest.approx(1.0)
    assert out.max() <= 0.05 + 1e-9


def test_apply_issuer_cap_raises_when_infeasible():
    # 3 names @ 5% cap can hold max 15%; can't sum to 100%.
    w = pd.Series({"A": 0.6, "B": 0.3, "C": 0.1})
    with pytest.raises(ValueError, match="infeasible"):
        apply_issuer_cap(w, cap=0.05)


def test_apply_issuer_cap_no_op_when_under_cap():
    w = pd.Series({"A": 0.04, "B": 0.03, "C": 0.93})
    # C is over cap; A and B absorb. Just verify total preserved.
    out = apply_issuer_cap(w, cap=0.95)
    assert out.sum() == pytest.approx(1.0)
    assert (out <= 0.95 + 1e-9).all()


def test_build_portfolio_end_to_end():
    inputs = _toy_inputs()
    scores = compute_momentum_scores(inputs)
    parent = pd.Series({"A": 100.0, "B": 200.0, "C": 50.0})
    pf = build_portfolio(scores, parent_mcap=parent, n=2, issuer_cap=None)
    assert set(pf.index) == {"A", "B"}  # C is the loser, gets dropped
    assert pf["weight"].sum() == pytest.approx(1.0)


# ---------- price-offset helper ----------


def test_month_end_closes_at_offset_picks_correct_row():
    idx = pd.date_range("2024-01-31", periods=14, freq="ME")
    monthly = pd.DataFrame(
        {"X": np.arange(100, 114, dtype=float)}, index=idx
    )
    rb = pd.Timestamp("2025-03-15")  # latest month-end <= rb is 2025-02-28 (idx=13)
    p1 = month_end_closes_at_offset(monthly, rb, 1)
    p7 = month_end_closes_at_offset(monthly, rb, 7)
    p13 = month_end_closes_at_offset(monthly, rb, 13)
    # idx 13 is the cutoff; 1 back = 12, 7 back = 6, 13 back = 0.
    assert p1["X"] == 112  # idx 12
    assert p7["X"] == 106  # idx 6
    assert p13["X"] == 100  # idx 0


def test_month_end_closes_at_offset_returns_nan_when_short():
    idx = pd.date_range("2025-01-31", periods=5, freq="ME")
    monthly = pd.DataFrame({"X": [100, 101, 102, 103, 104]}, index=idx)
    rb = pd.Timestamp("2025-05-31")
    p13 = month_end_closes_at_offset(monthly, rb, 13)
    assert np.isnan(p13["X"])
