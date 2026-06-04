# Agentic Finance — Distributional Shield

Companion notebooks for the Wiley book on AI agents: a finance-grade **distributional risk shield** that governs what AI agents are allowed to execute.

The core idea: **the LLM proposes, the shields decide.** An agent (or human) can suggest any trade, but every proposal must pass through a four-stage pipeline before anything executes:

```
unstructured instruction
        │
        ▼
typed Pydantic proposal          (semantic validity — fail closed on malformed input)
        │
        ▼
deterministic institutional shield   (restricted lists, notional / concentration /
        │                             gross-exposure / long-only / liquidity / DV01 limits)
        ▼
distributional financial shield      (Monte Carlo VaR, CVaR, tail-loss probability,
        │                             stress regimes, heavy tails)
        ▼
execute │ escalate │ reject          (full audit logging)
```

Everything is **self-contained and offline**: market data is synthetic, the LLM boundary is mocked, and no API keys or network access are needed. All notebooks run end-to-end with the pinned dependencies below.

## Notebooks

| Notebook | Sections | What it covers |
|---|---|---|
| [`notebooks/Agentic_Finance_Giant_Distributional_Shield_All_Examples.ipynb`](notebooks/Agentic_Finance_Giant_Distributional_Shield_All_Examples.ipynb) | 1–23 | Typed contracts → synthetic covariance model → deterministic baseline → distributional shield (VaR/CVaR/tail-prob) → worked examples (equity buy/sell, option delta, FX, credit, rates/DV01, basket rebalance) → stress regimes & heavy tails → model card → consolidated audit log → **multi-agent governance committee** (§21–23) |
| [`notebooks/Agentic_Finance_Giant_Distributional_Shield_Agent_Negotiator.ipynb`](notebooks/Agentic_Finance_Giant_Distributional_Shield_Agent_Negotiator.ipynb) | 1–25 | Everything above, plus the **agent negotiator shield** (§24–25): specialist negotiators that bargain a proposal down to the largest admissible size — but can never weaken the hard risk gates |
| [`notebooks/Agentic_Finance_Agentic_RAG_Distributional_Shield.ipynb`](notebooks/Agentic_Finance_Agentic_RAG_Distributional_Shield.ipynb) | 1–7 | **Agentic RAG layer**: builds a TF-IDF evidence corpus from the shield's audit logs in `data/`, retrieves top-k evidence per governance question, and produces grounded answers that fail closed (escalate) when evidence is weak — the RAG layer explains decisions, it never authorizes execution |

The second notebook is a strict superset of the first; sections 1–23 are identical in both. The RAG notebook consumes the audit CSVs the negotiator notebook produces (it reads them from `data/`, or from its own directory when run standalone).

### Multi-agent committee (sections 21–23)

Specialist agents — `ResearchAgent`, `TechnicalAgent`, `RiskAgent`, `ComplianceAgent` — vote on each proposal; a `PortfolioManagerAgent` reviews and aggregates, and a `MultiAgentShieldOrchestrator` routes every consensus through the distributional shield. The committee can recommend, challenge, escalate, or reject — it cannot bypass the shield. See [`docs/README_MULTIAGENT_EXTENSION.md`](docs/README_MULTIAGENT_EXTENSION.md).

### Agent negotiator shield (sections 24–25)

`ComplianceNegotiator`, `LiquidityNegotiator`, `DistributionalRiskNegotiator`, and `ExecutionNegotiator` run bounded negotiation rounds (typed offers, binding execution terms, bisection search for the largest survivable notional) orchestrated by `AgentNegotiatorShield`. Negotiation may only shrink or condition a trade — never relax a constraint — and the final proposal is re-evaluated by the shield, failing closed if still unsafe. See [`docs/README_AGENT_NEGOTIATOR_SHIELD.md`](docs/README_AGENT_NEGOTIATOR_SHIELD.md).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Run either notebook top to bottom (cells have sequential dependencies). Each run writes its CSV audit logs next to the notebook; those are gitignored — curated sample copies live in [`data/`](data/).

Requires Python ≥ 3.9 and Pydantic **v2** (the contracts use `model_validator`, `model_copy`, `model_config`).

## Sample data (`data/`)

All CSVs are synthetic simulation output from a verified end-to-end run of the Agent_Negotiator and Agentic_RAG notebooks. The Monte Carlo simulation uses fixed seeds, so results are reproducible on a given platform; exact metric values (VaR/CVaR/tail probability) can differ slightly across platforms and BLAS implementations — the decisions do not.

| File | Contents |
|---|---|
| `agentic_finance_giant_notebook_audit_log.csv` | One row per shield evaluation across all worked examples and stress/distribution variants (status, reason, VaR_95, CVaR_95, tail probability, exposure metrics, rules checked) |
| `agentic_finance_multiagent_decision_log.csv` | Final committee decision per proposal (consensus vs final status, shield verdict, risk metrics) |
| `agentic_finance_multiagent_opinion_log.csv` | One row per agent vote per proposal (role, agent_name, vote, confidence, reason, diagnostics) |
| `agentic_finance_agent_negotiator_shield_outcomes.csv` | Final negotiation outcome per session — six demo sessions incl. a $5M HY_CDS proposal negotiated down to ~$1M (original vs negotiated notional, status precedence, execution terms, shield verdict) |
| `agentic_finance_agent_negotiator_shield_offers.csv` | Per-round, per-negotiator offer log (votes, counteroffer notionals, binding terms, diagnostics) |
| `agentic_rag_corpus.csv` | RAG evidence corpus: architecture/governance notes plus one document per audit row, committee decision, negotiation outcome, and offer |
| `agentic_rag_queries.csv` | Example governance and execution questions (Q001–Q006) with expected focus |
| `agentic_rag_retrieval_log.csv` | Top-k retrieved evidence per query (doc ids, sources, TF-IDF cosine scores) |
| `agentic_rag_agent_opinion_log.csv` | Retriever / EvidenceQuality / DistributionalRisk / Compliance agent votes per query |
| `agentic_rag_answer_log.csv` | Final grounded answers with status (answer / escalate / reject), evidence ids, and the fail-closed weak-evidence gate |

## Repository layout

```
├── notebooks/    # the three executable notebooks (outputs included)
├── data/         # sample CSV audit logs + RAG corpus from a verified run
├── docs/         # per-extension design notes
├── requirements.txt
└── README.md
```

### Agentic RAG layer

`RetrievalEngine` (TF-IDF + cosine similarity) retrieves evidence from a corpus built out of the shield's own audit logs; an agentic review layer (Retriever / EvidenceQuality / DistributionalRisk / Compliance agents) votes on each answer. Weak or missing evidence triggers escalation rather than invention — the RAG layer grounds and explains shield decisions but cannot authorize execution. See [`docs/README_AGENTIC_RAG.md`](docs/README_AGENTIC_RAG.md).

## Design principles

- **Fail closed** — malformed input, unknown symbols, or any constraint breach yields reject/escalate, never silent execution.
- **Risk is a property of the resulting portfolio**, not the trade in isolation: shields evaluate `w_t + a_t`.
- **Agents advise, shields decide** — neither the committee nor the negotiators can override deterministic or distributional gates.
- **Everything is audited** — every evaluation, vote, offer, and verdict lands in a typed CSV log.
- **RAG explains, never executes** — answers must cite retrieved evidence and fail closed when it is weak.
