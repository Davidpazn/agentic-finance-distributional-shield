# Agentic Finance — Agentic RAG Notebook + CSVs

This package adds an agentic retrieval-augmented generation layer on top of the Distributional Shield framework.

## Files

- `notebooks/Agentic_Finance_Agentic_RAG_Distributional_Shield.ipynb` — the notebook; reads the shield evidence logs from `data/` (or from its own directory when run standalone).
- `data/agentic_rag_corpus.csv` — RAG corpus built from shield assumptions, audit rows, committee decisions, negotiator outcomes, and negotiation offers.
- `data/agentic_rag_queries.csv` — example governance and execution questions.
- `data/agentic_rag_retrieval_log.csv` — top-k retrieved evidence per query.
- `data/agentic_rag_agent_opinion_log.csv` — Retriever, EvidenceQuality, DistributionalRisk, and Compliance agent votes.
- `data/agentic_rag_answer_log.csv` — final grounded answers and execution/escalation/rejection status.

Running the notebook writes fresh copies of these next to it; curated sample copies are committed in `data/`.

## Design invariant

The RAG layer explains and grounds decisions; it does not authorize execution. If the retrieved evidence is weak, contradictory, or the shield rejects/escalates, the final answer fails closed.
