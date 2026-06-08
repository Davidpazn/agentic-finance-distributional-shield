"""Deterministic and distributional risk shields.

The shield is the deterministic core that sits between a (possibly stochastic)
proposal surface and the executor. ``DeterministicRiskShield`` is the minimal
restricted-list + notional gate; ``DistributionalRiskShield`` is the full
fail-closed gate chain the book calls B3: restricted list, desk notional, risk
universe, long-only, gross exposure, concentration, liquidity/ADV, then the
Monte-Carlo VaR / CVaR / tail-probability checks. Every branch fails closed, and
the catch-all at the end turns any unexpected error into a reject rather than a
silent allow. Ported in semantics from the giant distributional-shield notebook.
"""
from __future__ import annotations

import math
import uuid
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel

from .contracts import (
    AssetClass,
    DistributionalRiskReport,
    DistributionalShieldDecision,
    PortfolioState,
    ProposedAction,
    RiskPolicy,
)
from .market import nearest_psd


class DeterministicShieldDecision(BaseModel):
    model_config = {"extra": "forbid"}

    status: Literal["allow", "reject"]
    reason: str
    checked_rules: list[str]


class DeterministicRiskShield:
    """Minimal deterministic gate: restricted list then desk notional limit."""

    def __init__(self, desk_risk_limit: float, restricted_list: list[str]):
        self.desk_risk_limit = desk_risk_limit
        # Normalized once so case or whitespace variants cannot slip past the check.
        self.restricted_list = {entry.strip().upper() for entry in restricted_list}

    def evaluate(self, proposal: ProposedAction) -> DeterministicShieldDecision:
        checked_rules: list[str] = []

        checked_rules.append("RESTRICTED_LIST_CHECK")
        if proposal.symbol in self.restricted_list:
            return DeterministicShieldDecision(
                status="reject",
                reason=f"Symbol {proposal.symbol} is restricted.",
                checked_rules=checked_rules,
            )

        checked_rules.append("NOTIONAL_LIMIT_CHECK")
        if proposal.notional_value > self.desk_risk_limit:
            return DeterministicShieldDecision(
                status="reject",
                reason=(
                    f"Notional ${proposal.notional_value:,.2f} exceeds desk limit "
                    f"${self.desk_risk_limit:,.2f}."
                ),
                checked_rules=checked_rules,
            )

        return DeterministicShieldDecision(
            status="allow",
            reason="All deterministic checks passed.",
            checked_rules=checked_rules,
        )


class DistributionalRiskShield:
    """The full B3 fail-closed gate chain over a typed proposal and portfolio."""

    def __init__(
        self,
        policy: RiskPolicy,
        n_simulations: int = 25_000,
        random_seed: int = 42,
        distribution: Literal["normal", "student_t"] = "normal",
        student_t_df: int = 5,
        stress_multiplier: float = 1.0,
        stress_label: str = "base",
    ):
        self.policy = policy
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.distribution = distribution
        self.student_t_df = student_t_df
        self.stress_multiplier = stress_multiplier
        self.stress_label = stress_label

    def _proposal_to_exposure_vector(
        self,
        proposal: ProposedAction,
        portfolio_state: PortfolioState,
    ) -> tuple[np.ndarray, float, Optional[float]]:
        symbols = portfolio_state.symbols
        if proposal.symbol not in symbols:
            raise ValueError(f"Symbol {proposal.symbol} is not in the portfolio risk universe.")

        exposure = np.array(
            [portfolio_state.current_exposure.get(symbol, 0.0) for symbol in symbols],
            dtype=float,
        )
        index = symbols.index(proposal.symbol)
        signed_exposure_change = proposal.signed_notional()

        # Options are converted into delta-equivalent exposure.
        if proposal.asset_class == AssetClass.OPTION and proposal.delta is not None:
            signed_exposure_change = proposal.signed_notional() * proposal.delta

        exposure[index] = exposure[index] + signed_exposure_change

        adv = portfolio_state.adv_notional.get(proposal.symbol)
        trade_adv_fraction = None
        if adv is not None and adv > 0:
            trade_adv_fraction = proposal.notional_value / adv

        return exposure, signed_exposure_change, trade_adv_fraction

    def _draw_returns(self, covariance: np.ndarray, n_assets: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        covariance = nearest_psd(covariance * self.stress_multiplier)
        mean_returns = np.zeros(n_assets)

        if self.distribution == "normal":
            return rng.multivariate_normal(
                mean=mean_returns,
                cov=covariance,
                size=self.n_simulations,
            )

        gaussian_draws = rng.multivariate_normal(
            mean=mean_returns,
            cov=covariance,
            size=self.n_simulations,
        )
        # A raw multivariate-t with nu degrees of freedom has covariance nu/(nu-2)
        # times its scale matrix. The correction below keeps the simulated covariance
        # equal to the input matrix, so regimes differ in TAIL SHAPE, not in variance.
        if self.student_t_df <= 2:
            raise ValueError("student_t_df must be greater than 2 for a finite covariance.")
        scale = np.sqrt(self.student_t_df / rng.chisquare(self.student_t_df, size=self.n_simulations))
        variance_correction = math.sqrt((self.student_t_df - 2) / self.student_t_df)
        return gaussian_draws * (scale[:, None] * variance_correction)

    def simulate_losses_for_plot(
        self,
        proposal: ProposedAction,
        portfolio_state: PortfolioState,
    ) -> np.ndarray:
        covariance = np.array(portfolio_state.covariance_matrix, dtype=float)
        exposure_after_trade, _, _ = self._proposal_to_exposure_vector(
            proposal=proposal,
            portfolio_state=portfolio_state,
        )
        simulated_returns = self._draw_returns(covariance, len(portfolio_state.symbols))
        simulated_pnl = simulated_returns @ exposure_after_trade
        return -simulated_pnl

    def _simulate_loss_distribution(
        self,
        proposal: ProposedAction,
        portfolio_state: PortfolioState,
    ) -> DistributionalRiskReport:
        simulated_losses = self.simulate_losses_for_plot(
            proposal=proposal,
            portfolio_state=portfolio_state,
        )
        exposure_after_trade, _, trade_adv_fraction = self._proposal_to_exposure_vector(
            proposal=proposal,
            portfolio_state=portfolio_state,
        )

        var_95 = float(np.quantile(simulated_losses, 0.95))
        tail_losses = simulated_losses[simulated_losses >= var_95]
        cvar_95 = float(tail_losses.mean()) if len(tail_losses) else var_95
        probability_loss_exceeds_limit = float(np.mean(simulated_losses > self.policy.loss_limit))
        gross_exposure = float(np.sum(np.abs(exposure_after_trade)))
        max_symbol_weight_after = float(np.max(np.abs(exposure_after_trade)) / portfolio_state.current_value)
        gross_exposure_multiple_after = gross_exposure / portfolio_state.current_value

        return DistributionalRiskReport(
            var_95=var_95,
            cvar_95=cvar_95,
            probability_loss_exceeds_limit=probability_loss_exceeds_limit,
            expected_loss=float(simulated_losses.mean()),
            loss_std=float(simulated_losses.std(ddof=1)),
            simulated_losses=self.n_simulations,
            max_symbol_weight_after=max_symbol_weight_after,
            gross_exposure_multiple_after=gross_exposure_multiple_after,
            trade_adv_fraction=trade_adv_fraction,
            stress_label=self.stress_label,
            diagnostics={
                "distribution": self.distribution,
                "student_t_df": self.student_t_df if self.distribution == "student_t" else None,
                "stress_multiplier": self.stress_multiplier,
                # The governing limits are recorded so every audit row is interpretable
                # on its own: a probability or CVaR only has meaning next to its limit.
                "policy_name": self.policy.name,
                "loss_limit": self.policy.loss_limit,
                "max_var_95": self.policy.max_var_95,
                "max_cvar_95": self.policy.max_cvar_95,
                "max_symbol_weight": self.policy.max_symbol_weight,
                "max_gross_exposure_multiple": self.policy.max_gross_exposure_multiple,
                "max_adv_fraction": self.policy.max_adv_fraction,
            },
        )

    def evaluate(
        self,
        proposal: ProposedAction,
        portfolio_state: PortfolioState,
    ) -> DistributionalShieldDecision:
        audit_id = str(uuid.uuid4())[:8]
        evaluated_rules: list[str] = []
        payload = proposal.model_dump(mode="json")

        try:
            evaluated_rules.append("RESTRICTED_LIST_CHECK")
            restricted = {entry.strip().upper() for entry in self.policy.restricted_list}
            if proposal.symbol in restricted:
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status="reject",
                    reason=f"Symbol {proposal.symbol} is restricted.",
                    deterministic_rules=evaluated_rules,
                    proposal_payload=payload,
                )

            evaluated_rules.append("NOTIONAL_LIMIT_CHECK")
            if proposal.notional_value > self.policy.desk_risk_limit:
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status="reject",
                    reason=(
                        f"Notional ${proposal.notional_value:,.2f} exceeds desk limit "
                        f"${self.policy.desk_risk_limit:,.2f}."
                    ),
                    deterministic_rules=evaluated_rules,
                    proposal_payload=payload,
                )

            evaluated_rules.append("RISK_UNIVERSE_CHECK")
            if proposal.symbol not in portfolio_state.symbols:
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status="reject",
                    reason=f"Symbol {proposal.symbol} is outside the approved risk universe.",
                    deterministic_rules=evaluated_rules,
                    proposal_payload=payload,
                )

            exposure_after_trade, _, trade_adv_fraction = self._proposal_to_exposure_vector(
                proposal=proposal,
                portfolio_state=portfolio_state,
            )

            evaluated_rules.append("LONG_ONLY_CHECK")
            # Only the traded symbol's resulting exposure is checked: a pre-existing
            # short elsewhere in the book must not veto an unrelated long-only trade.
            traded_index = portfolio_state.symbols.index(proposal.symbol)
            if self.policy.long_only and exposure_after_trade[traded_index] < -1e-8:
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status="reject",
                    reason=(
                        f"{proposal.side} {proposal.symbol} would take {proposal.symbol} exposure to "
                        f"${exposure_after_trade[traded_index]:,.0f} under a long-only policy."
                    ),
                    deterministic_rules=evaluated_rules,
                    proposal_payload=payload,
                )

            gross_exposure_multiple = float(np.sum(np.abs(exposure_after_trade)) / portfolio_state.current_value)
            max_symbol_weight = float(np.max(np.abs(exposure_after_trade)) / portfolio_state.current_value)

            evaluated_rules.append("GROSS_EXPOSURE_CHECK")
            if gross_exposure_multiple > self.policy.max_gross_exposure_multiple:
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status="reject",
                    reason=(
                        f"Gross exposure multiple {gross_exposure_multiple:.2f} exceeds "
                        f"limit {self.policy.max_gross_exposure_multiple:.2f}."
                    ),
                    deterministic_rules=evaluated_rules,
                    proposal_payload=payload,
                )

            evaluated_rules.append("CONCENTRATION_CHECK")
            if max_symbol_weight > self.policy.max_symbol_weight:
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status="reject",
                    reason=(
                        f"Single-symbol exposure {max_symbol_weight:.2%} exceeds "
                        f"limit {self.policy.max_symbol_weight:.2%}."
                    ),
                    deterministic_rules=evaluated_rules,
                    proposal_payload=payload,
                )

            evaluated_rules.append("LIQUIDITY_ADV_CHECK")
            if trade_adv_fraction is None:
                # Fail closed: a tradable symbol with no usable ADV estimate cannot
                # have its liquidity verified, so it is never silently waved through.
                status = "escalate" if self.policy.escalate_on_liquidity else "reject"
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status=status,
                    reason=f"No usable ADV estimate for {proposal.symbol}; liquidity cannot be verified.",
                    deterministic_rules=evaluated_rules,
                    proposal_payload=payload,
                )
            if trade_adv_fraction > self.policy.max_adv_fraction:
                status = "escalate" if self.policy.escalate_on_liquidity else "reject"
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status=status,
                    reason=(
                        f"Trade consumes {trade_adv_fraction:.2%} of ADV, above "
                        f"limit {self.policy.max_adv_fraction:.2%}."
                    ),
                    deterministic_rules=evaluated_rules,
                    proposal_payload=payload,
                )

            evaluated_rules.extend([
                "DISTRIBUTIONAL_VAR_CHECK",
                "DISTRIBUTIONAL_CVAR_CHECK",
                "TAIL_PROBABILITY_CHECK",
            ])
            risk_report = self._simulate_loss_distribution(
                proposal=proposal,
                portfolio_state=portfolio_state,
            )

            evaluated_rules.append("METRIC_SANITY_CHECK")
            if not all(math.isfinite(value) for value in (
                risk_report.var_95, risk_report.cvar_95, risk_report.probability_loss_exceeds_limit,
            )):
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status="reject",
                    reason="Distributional risk metrics are not finite; failing closed.",
                    deterministic_rules=evaluated_rules,
                    distributional_report=risk_report,
                    proposal_payload=payload,
                )

            # Hard risk reject first. This avoids routing a clearly unacceptable tail risk to soft PM override.
            if (
                self.policy.reject_on_tail_probability
                and risk_report.probability_loss_exceeds_limit > self.policy.max_probability_loss_exceeds_limit
            ):
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status="reject",
                    reason=(
                        f"Tail-loss probability {risk_report.probability_loss_exceeds_limit:.2%} exceeds "
                        f"limit {self.policy.max_probability_loss_exceeds_limit:.2%}."
                    ),
                    deterministic_rules=evaluated_rules,
                    distributional_report=risk_report,
                    proposal_payload=payload,
                )

            # Soft risk escalation after hard rejects.
            if risk_report.cvar_95 > self.policy.max_cvar_95:
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status="escalate",
                    reason=(
                        f"CVaR 95% ${risk_report.cvar_95:,.2f} exceeds "
                        f"limit ${self.policy.max_cvar_95:,.2f}."
                    ),
                    deterministic_rules=evaluated_rules,
                    distributional_report=risk_report,
                    proposal_payload=payload,
                )

            if risk_report.var_95 > self.policy.max_var_95:
                return DistributionalShieldDecision(
                    audit_id=audit_id,
                    status="escalate",
                    reason=(
                        f"VaR 95% ${risk_report.var_95:,.2f} exceeds "
                        f"limit ${self.policy.max_var_95:,.2f}."
                    ),
                    deterministic_rules=evaluated_rules,
                    distributional_report=risk_report,
                    proposal_payload=payload,
                )

            return DistributionalShieldDecision(
                audit_id=audit_id,
                status="allow",
                reason="All deterministic and distributional checks passed.",
                deterministic_rules=evaluated_rules,
                distributional_report=risk_report,
                proposal_payload=payload,
            )

        except Exception as error:
            # Deliberate catch-all: any unexpected failure inside the shield must fail
            # closed. The exception type is surfaced so genuine bugs stay visible in
            # the audit trail instead of hiding behind a generic message.
            return DistributionalShieldDecision(
                audit_id=audit_id,
                status="reject",
                reason=f"Shield failed closed on {type(error).__name__}: {error}",
                deterministic_rules=evaluated_rules,
                proposal_payload=payload,
            )
