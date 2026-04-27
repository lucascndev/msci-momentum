"""MSCI Momentum index replication."""

from msci_momentum.momentum import compute_momentum_scores, build_inputs_for_universe
from msci_momentum.portfolio import build_portfolio

__all__ = [
    "build_inputs_for_universe",
    "build_portfolio",
    "compute_momentum_scores",
]
