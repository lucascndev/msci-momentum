"""MSCI Momentum scoring (methodology section 2.2).

Pipeline (per rebalance date T):

  1. 6m raw return  = P_{T-1} / P_{T-7}  - 1  - rf_6m
     12m raw return = P_{T-1} / P_{T-13} - 1  - rf_12m

  2. Risk-adjust by sigma = annualized stdev of weekly local returns over 3y.

  3. z-score each (cross-section), average 50/50, z-score the combined value
     again, and winsorize at ±3.

  4. Map Z to MomentumScore: 1 + Z if Z > 0, else 1 / (1 - Z).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from msci_momentum.data import (
    fetch_monthly_closes,
    fetch_weekly_closes,
    month_end_closes_at_offset,
)
from msci_momentum.risk_free import horizon_rate


@dataclass
class MomentumInputs:
    """Per-stock inputs needed to compute Momentum Score for one rebalance date."""

    p_t_minus_1: pd.Series   # last month-end close before T
    p_t_minus_7: pd.Series   # 7 month-ends back
    p_t_minus_13: pd.Series  # 13 month-ends back
    sigma_annual: pd.Series  # annualized stdev of weekly local returns over 3y
    rf_6m: float             # 6-month risk-free rate (decimal)
    rf_12m: float            # 12-month risk-free rate (decimal)


def annualized_weekly_volatility(weekly_closes: pd.DataFrame) -> pd.Series:
    """Annualized stdev of weekly returns. Drops series with < 52 obs."""
    returns = weekly_closes.pct_change(fill_method=None)
    counts = returns.count()
    sigma = returns.std(ddof=1) * np.sqrt(52.0)
    sigma[counts < 52] = np.nan
    sigma.name = "sigma_annual"
    return sigma


def _zscore(x: pd.Series) -> pd.Series:
    mu = x.mean(skipna=True)
    sd = x.std(ddof=1, skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def _msci_score_from_z(z: pd.Series) -> pd.Series:
    """Z>0 -> 1+Z; Z<=0 -> 1/(1-Z). Always positive, monotone in Z."""
    out = pd.Series(np.nan, index=z.index, dtype=float)
    pos = z > 0
    neg = ~pos & z.notna()
    out.loc[pos] = 1.0 + z.loc[pos]
    out.loc[neg] = 1.0 / (1.0 - z.loc[neg])
    return out


def compute_momentum_scores(
    inputs: MomentumInputs,
    use_only_6m: bool = False,
) -> pd.DataFrame:
    """Run the full MSCI scoring pipeline.

    Returns a DataFrame indexed by ticker with columns:
      raw_6m, raw_12m, riskadj_6m, riskadj_12m,
      z_6m, z_12m, combined, z_unwinsorized, z_winsorized, momentum_score
    """
    raw_6m = (inputs.p_t_minus_1 / inputs.p_t_minus_7) - 1.0 - inputs.rf_6m
    raw_12m = (inputs.p_t_minus_1 / inputs.p_t_minus_13) - 1.0 - inputs.rf_12m

    sigma = inputs.sigma_annual.replace(0.0, np.nan)
    riskadj_6m = raw_6m / sigma
    riskadj_12m = raw_12m / sigma

    if use_only_6m:
        # Ad-hoc rebalance path (Appendix III): only 6m used.
        combined = riskadj_6m.copy()
        z6 = _zscore(riskadj_6m)
        z12 = pd.Series(np.nan, index=riskadj_12m.index)
    else:
        z6 = _zscore(riskadj_6m)
        # If 12m is missing, fall back to 6m only for that name (per spec):
        z12_full = _zscore(riskadj_12m)
        z12 = z12_full
        combined = pd.Series(np.nan, index=z6.index, dtype=float)
        both = z6.notna() & z12_full.notna()
        only6 = z6.notna() & z12_full.isna()
        combined.loc[both] = 0.5 * z6.loc[both] + 0.5 * z12_full.loc[both]
        combined.loc[only6] = z6.loc[only6]

    z_unwinsorized = _zscore(combined)
    z_winsorized = z_unwinsorized.clip(lower=-3.0, upper=3.0)
    score = _msci_score_from_z(z_winsorized)

    return pd.DataFrame(
        {
            "raw_6m": raw_6m,
            "raw_12m": raw_12m,
            "riskadj_6m": riskadj_6m,
            "riskadj_12m": riskadj_12m,
            "z_6m": z6,
            "z_12m": z12,
            "combined": combined,
            "z_unwinsorized": z_unwinsorized,
            "z_winsorized": z_winsorized,
            "momentum_score": score,
        }
    )


def build_inputs_for_universe(
    tickers: list[str],
    rebalance_date: pd.Timestamp,
    country: str = "USA",
    use_cache: bool = True,
) -> MomentumInputs:
    """Fetch all data and assemble MomentumInputs for ``tickers`` at T."""
    monthly = fetch_monthly_closes(tickers, rebalance_date, use_cache=use_cache)
    weekly = fetch_weekly_closes(tickers, rebalance_date, use_cache=use_cache)

    p1 = month_end_closes_at_offset(monthly, rebalance_date, 1)
    p7 = month_end_closes_at_offset(monthly, rebalance_date, 7)
    p13 = month_end_closes_at_offset(monthly, rebalance_date, 13)
    sigma = annualized_weekly_volatility(weekly)

    rf_6m = horizon_rate(country, pd.Timestamp(rebalance_date), 6)
    rf_12m = horizon_rate(country, pd.Timestamp(rebalance_date), 12)

    return MomentumInputs(
        p_t_minus_1=p1,
        p_t_minus_7=p7,
        p_t_minus_13=p13,
        sigma_annual=sigma,
        rf_6m=rf_6m,
        rf_12m=rf_12m,
    )
