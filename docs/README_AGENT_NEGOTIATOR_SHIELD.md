# Agentic Finance — Agent Negotiator Shield Extension

This extension adds an agent negotiator shielding layer on top of the multi-agent distributional shield.

## Main notebook

- `notebooks/Agentic_Finance_Giant_Distributional_Shield_Agent_Negotiator.ipynb` — the negotiator layer is in **sections 24–25**; sections 1–23 are the base distributional shield and multi-agent committee.

## New CSV audit logs

- `agentic_finance_agent_negotiator_shield_outcomes.csv`
- `agentic_finance_agent_negotiator_shield_offers.csv`

Running the notebook writes these next to it; sample copies are committed in `data/`.

## What was added

The negotiator shield introduces typed negotiation objects and specialist negotiators:

1. `ComplianceNegotiator`: hard-blocks restricted or non-approved symbols.
2. `LiquidityNegotiator`: counteroffers smaller notional size based on ADV limits.
3. `DistributionalRiskNegotiator`: searches for the largest notional that survives VaR/CVaR/tail-risk constraints.
4. `ExecutionNegotiator`: attaches execution terms such as staged VWAP, child orders, participation limits, and kill switches.
5. `AgentNegotiatorShield`: orchestrates negotiation rounds and submits the final negotiated proposal to the distributional shield.

The negotiation layer can reduce or condition a trade, but it cannot bypass deterministic or distributional risk gates.
