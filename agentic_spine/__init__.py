"""Agentic-finance distributional-shield reference library.

This package factors the typed contracts and the deterministic /
distributional shields out of the companion notebooks
(``Agentic_Finance_Giant_Distributional_Shield_All_Examples.ipynb``) into an
importable module so that experiments and tests can reuse *exactly* the same
gate semantics rather than re-implementing them.

The execution-spine ablation (``experiments/execution_spine.py``) is built
entirely on top of the objects exported here, so that the B3 ("book
architecture") system and the notebook share one source of truth for what a
``ProposedAction`` is and what the shield does to it.
"""
from __future__ import annotations

from .contracts import (
    ActionType,
    AssetClass,
    BasketAction,
    DistributionalRiskReport,
    DistributionalShieldDecision,
    PortfolioState,
    ProposedAction,
    RiskPolicy,
)
from .market import build_covariance, core_policy, make_base_portfolio, nearest_psd
from .shields import (
    DeterministicRiskShield,
    DeterministicShieldDecision,
    DistributionalRiskShield,
)

__all__ = [
    "ActionType",
    "AssetClass",
    "ProposedAction",
    "BasketAction",
    "PortfolioState",
    "RiskPolicy",
    "DistributionalRiskReport",
    "DistributionalShieldDecision",
    "nearest_psd",
    "build_covariance",
    "make_base_portfolio",
    "core_policy",
    "DeterministicShieldDecision",
    "DeterministicRiskShield",
    "DistributionalRiskShield",
]
