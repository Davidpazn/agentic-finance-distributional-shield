"""Typed contracts shared by the shields, the notebooks, and the experiments.

These classes are the *semantic-validity* boundary of the architecture: a
language model (or any proposal surface) may emit free text, but only a
well-typed :class:`ProposedAction` can ever reach a shield, and only an
admissible action can reach the executor. The definitions here are ported
verbatim (in semantics) from the giant distributional-shield notebook so that
the execution-spine experiment exercises the same parse/validate behaviour the
book describes.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class ActionType(str, Enum):
    EXECUTE_TRADE = "execute_trade"
    REBALANCE_PORTFOLIO = "rebalance_portfolio"
    HEDGE_EXPOSURE = "hedge_exposure"


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    FX = "fx"
    OPTION = "option"
    BOND = "bond"
    CDS = "cds"
    COMMODITY = "commodity"
    RISK_FACTOR = "risk_factor"


class ProposedAction(BaseModel):
    """Semantic validity: a typed candidate action produced by the LLM boundary."""

    model_config = {"extra": "forbid"}

    action_type: ActionType = ActionType.EXECUTE_TRADE
    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    side: Literal["BUY", "SELL"]
    notional_value: float = Field(ge=0.0)
    quantity: Optional[float] = Field(default=None, ge=0.0)
    price: Optional[float] = Field(default=None, ge=0.0)
    delta: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    dv01: Optional[float] = None
    currency: str = "USD"
    sector: Optional[str] = None
    rationale: str

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Symbols are compared against restricted lists and the risk universe;
        normalizing here closes case/whitespace bypasses such as 'gme' or 'GME '."""
        return value.strip().upper()

    @model_validator(mode="after")
    def validate_finite_numbers(self) -> "ProposedAction":
        for field_name in ("notional_value", "quantity", "price", "delta", "dv01"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be a finite number.")
        return self

    def signed_notional(self) -> float:
        sign = 1.0 if self.side == "BUY" else -1.0
        return sign * self.notional_value


class BasketAction(BaseModel):
    """A multi-trade proposal, useful for portfolio rebalancing or hedge baskets."""

    model_config = {"extra": "forbid"}

    name: str
    trades: list[ProposedAction]
    rationale: str


class PortfolioState(BaseModel):
    """Portfolio state required by deterministic and distributional checks."""

    model_config = {"extra": "forbid"}

    current_value: float = Field(gt=0.0)
    current_exposure: dict[str, float]
    covariance_matrix: list[list[float]]
    symbols: list[str]
    symbol_sector: dict[str, str] = Field(default_factory=dict)
    symbol_currency: dict[str, str] = Field(default_factory=dict)
    adv_notional: dict[str, float] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def validate_shape(self) -> "PortfolioState":
        n = len(self.symbols)
        if len(self.covariance_matrix) != n:
            raise ValueError("Covariance matrix row count must match number of symbols.")
        for row in self.covariance_matrix:
            if len(row) != n:
                raise ValueError("Covariance matrix column count must match number of symbols.")
        if len(set(self.symbols)) != n:
            raise ValueError("Portfolio symbols must be unique.")
        unknown_exposures = set(self.current_exposure) - set(self.symbols)
        if unknown_exposures:
            raise ValueError(f"Exposure keys outside the symbol universe: {sorted(unknown_exposures)}.")
        matrix = np.array(self.covariance_matrix, dtype=float)
        if not np.allclose(matrix, matrix.T, atol=1e-12):
            raise ValueError("Covariance matrix must be symmetric.")
        return self


class RiskPolicy(BaseModel):
    """Institutional and financial admissibility limits."""

    model_config = {"extra": "forbid"}

    name: str = "Core risk policy"
    desk_risk_limit: float = Field(gt=0.0)
    restricted_list: list[str] = Field(default_factory=list)
    max_var_95: float = Field(gt=0.0)
    max_cvar_95: float = Field(gt=0.0)
    loss_limit: float = Field(gt=0.0)
    max_probability_loss_exceeds_limit: float = Field(ge=0.0, le=1.0)
    max_symbol_weight: float = Field(gt=0.0)
    max_gross_exposure_multiple: float = Field(gt=0.0)
    max_adv_fraction: float = Field(gt=0.0)
    long_only: bool = False
    reject_on_tail_probability: bool = True
    escalate_on_liquidity: bool = True


class DistributionalRiskReport(BaseModel):
    model_config = {"extra": "forbid"}

    var_95: float
    cvar_95: float
    probability_loss_exceeds_limit: float = Field(ge=0.0, le=1.0)
    expected_loss: float
    loss_std: float
    simulated_losses: int
    max_symbol_weight_after: float
    gross_exposure_multiple_after: float
    trade_adv_fraction: Optional[float] = None
    stress_label: str = "base"
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class DistributionalShieldDecision(BaseModel):
    """Canonical audit object: decision, reasons, rules, and risk report."""

    model_config = {"extra": "forbid"}

    audit_id: str
    status: Literal["allow", "reject", "escalate"]
    reason: str
    deterministic_rules: list[str]
    distributional_report: Optional[DistributionalRiskReport] = None
    proposal_payload: dict[str, Any]
