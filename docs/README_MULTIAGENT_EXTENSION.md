# Multi-Agent Extension

This extension grows the original distributional-shield notebook into a multi-agent governance architecture.

## Where to find it

The multi-agent layer lives in **sections 21–23** of both notebooks in `notebooks/`:

- `Agentic_Finance_Giant_Distributional_Shield_All_Examples.ipynb` (sections 1–23)
- `Agentic_Finance_Giant_Distributional_Shield_Agent_Negotiator.ipynb` (sections 1–25, adds the negotiator layer)

## New sections
- Multi-agent typed contracts: `AgentRole`, `AgentVote`, `AgentOpinion`, `MultiAgentDecision`
- Specialist agents (class name → `agent_name` used in the CSV logs):
  - `ResearchAgent` → `FundamentalResearchAgent`
  - `TechnicalAgent` → `TechnicalTimingAgent`
  - `RiskAgent` → `DistributionalRiskAgent`
  - `ComplianceAgent` → `ComplianceAgent`
  - `PortfolioManagerAgent` — reviews and aggregates the committee's votes; it does not appear as a voting row in the opinion log.
- `MultiAgentShieldOrchestrator`, which aggregates agent votes and still routes every proposal through the distributional shield.

## New CSV outputs
- `agentic_finance_multiagent_decision_log.csv`
- `agentic_finance_multiagent_opinion_log.csv`

Running the notebook writes these next to it; sample copies are committed in `data/`.

## Design principle
The multi-agent committee can recommend, challenge, escalate, or reject, but it cannot bypass the deterministic and distributional risk shield. The shield remains the executable control layer.
