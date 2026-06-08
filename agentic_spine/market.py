"""Synthetic market, base portfolio, and the core risk policy.

The covariance model and the base portfolio are the same synthetic multi-asset
teaching book used in the notebook. ``core_policy`` returns the exact limits the
book chapter and the committed audit log pin for the execution-spine packet:
$1M desk notional, 45% single-name, 1.60 gross, 15% ADV, 5% tail-probability,
VaR95 $75k, CVaR95 $95k, long-only, ``GME`` restricted.
"""
from __future__ import annotations

import numpy as np

from .contracts import PortfolioState, RiskPolicy

# Canonical symbol universe for the synthetic book.
SYMBOLS: list[str] = [
    "AAPL", "MSFT", "TSLA", "NVDA", "JPM", "TLT", "EURUSD", "OIL", "HY_CDS", "US10Y",
]


def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Return a numerically positive semidefinite version of a symmetric matrix."""
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, epsilon)
    return eigenvectors @ np.diag(clipped) @ eigenvectors.T


def build_covariance(symbols: list[str] = SYMBOLS) -> np.ndarray:
    """Synthetic daily covariance matrix for a multi-asset teaching example."""
    vol = {
        "AAPL": 0.020,
        "MSFT": 0.018,
        "TSLA": 0.038,
        "NVDA": 0.034,
        "JPM": 0.021,
        "TLT": 0.012,
        "EURUSD": 0.006,
        "OIL": 0.027,
        "HY_CDS": 0.010,
        "US10Y": 0.007,
    }
    n = len(symbols)
    corr = np.eye(n)
    idx = {symbol: i for i, symbol in enumerate(symbols)}

    equity_names = ["AAPL", "MSFT", "TSLA", "NVDA", "JPM"]
    for a in equity_names:
        for b in equity_names:
            if a in idx and b in idx and a != b:
                corr[idx[a], idx[b]] = 0.52

    special_pairs = {
        ("TLT", "AAPL"): -0.20,
        ("TLT", "MSFT"): -0.18,
        ("TLT", "TSLA"): -0.15,
        ("TLT", "NVDA"): -0.16,
        ("TLT", "JPM"): -0.22,
        ("HY_CDS", "AAPL"): -0.35,
        ("HY_CDS", "MSFT"): -0.32,
        ("HY_CDS", "TSLA"): -0.40,
        ("HY_CDS", "NVDA"): -0.38,
        ("HY_CDS", "JPM"): -0.45,
        ("OIL", "JPM"): 0.25,
        ("OIL", "AAPL"): 0.18,
        ("OIL", "MSFT"): 0.12,
        ("EURUSD", "AAPL"): 0.08,
        ("EURUSD", "MSFT"): 0.06,
        ("US10Y", "TLT"): -0.60,
        ("US10Y", "JPM"): 0.20,
        ("US10Y", "HY_CDS"): 0.15,
    }
    for (a, b), value in special_pairs.items():
        if a in idx and b in idx:
            corr[idx[a], idx[b]] = value
            corr[idx[b], idx[a]] = value

    vols = np.array([vol[symbol] for symbol in symbols])
    covariance = np.diag(vols) @ corr @ np.diag(vols)
    return nearest_psd(covariance)


def make_base_portfolio() -> PortfolioState:
    """The synthetic multi-asset book the execution-spine packet trades against."""
    covariance = build_covariance(SYMBOLS)
    return PortfolioState(
        current_value=2_500_000.0,
        current_exposure={
            "AAPL": 650_000.0,
            "MSFT": 450_000.0,
            "TSLA": 250_000.0,
            "NVDA": 200_000.0,
            "JPM": 200_000.0,
            "TLT": 300_000.0,
            "EURUSD": 100_000.0,
            "OIL": 75_000.0,
            "HY_CDS": -100_000.0,
            "US10Y": 0.0,
        },
        covariance_matrix=covariance.tolist(),
        symbols=list(SYMBOLS),
        symbol_sector={
            "AAPL": "Technology",
            "MSFT": "Technology",
            "TSLA": "Consumer Discretionary",
            "NVDA": "Technology",
            "JPM": "Financials",
            "TLT": "Rates",
            "EURUSD": "FX",
            "OIL": "Commodity",
            "HY_CDS": "Credit",
            "US10Y": "Rates",
        },
        symbol_currency={
            "AAPL": "USD",
            "MSFT": "USD",
            "TSLA": "USD",
            "NVDA": "USD",
            "JPM": "USD",
            "TLT": "USD",
            "EURUSD": "EURUSD",
            "OIL": "USD",
            "HY_CDS": "USD",
            "US10Y": "USD",
        },
        adv_notional={
            "AAPL": 8_000_000_000.0,
            "MSFT": 6_000_000_000.0,
            "TSLA": 10_000_000_000.0,
            "NVDA": 12_000_000_000.0,
            "JPM": 3_000_000_000.0,
            "TLT": 2_000_000_000.0,
            "EURUSD": 60_000_000_000.0,
            "OIL": 1_000_000_000.0,
            "HY_CDS": 25_000_000.0,
            "US10Y": 4_000_000_000.0,
        },
        notes="Synthetic multi-asset portfolio for the agentic-finance execution spine.",
    )


def core_policy() -> RiskPolicy:
    """The pinned risk policy for the execution-spine packet.

    These are the limits recorded in the committed giant-notebook audit log,
    with ``long_only`` enabled as stated in the experiment README.
    """
    return RiskPolicy(
        name="Core teaching policy",
        desk_risk_limit=1_000_000.0,
        restricted_list=["GME", "AMC"],
        max_var_95=75_000.0,
        max_cvar_95=95_000.0,
        loss_limit=80_000.0,
        max_probability_loss_exceeds_limit=0.05,
        max_symbol_weight=0.45,
        max_gross_exposure_multiple=1.60,
        max_adv_fraction=0.15,
        long_only=True,
    )
