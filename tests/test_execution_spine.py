"""Assertions for the execution-spine ablation.

These tests pin every load-bearing claim the book chapter makes about the
fixed packet: the headline table is reproduced exactly, the integer columns do
not depend on the Monte-Carlo seed (because the unsafe cases breach
deterministic gates), prompt injection in a rationale has no authority, the
malformed payload fails schema validation, and only B3 produces a replayable
audit trail.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest
from pydantic import ValidationError

from agentic_spine import (
    DistributionalRiskShield,
    ProposedAction,
    core_policy,
    make_base_portfolio,
)
from experiments.execution_spine import (
    DEFAULT_N_SIMS,
    DEFAULT_SEED,
    EXPECTED_HEADLINE,
    INJECTION_RATIONALE,
    audit_frame,
    build_packet,
    check_against_book,
    clopper_pearson,
    headline_frame,
    main,
    metrics_frame,
    replay_b3,
    run_experiment,
    run_identifier,
)

SYSTEMS = ("B0", "B1", "B2", "B3")


@pytest.fixture(scope="module")
def result():
    return run_experiment(seed=DEFAULT_SEED, n_sims=DEFAULT_N_SIMS)


# --------------------------------------------------------------------------- #
# Packet composition
# --------------------------------------------------------------------------- #
def test_packet_composition():
    packet = build_packet()
    assert len(packet) == 8
    labels = [c.label for c in packet]
    assert labels.count("safe") == 4
    assert labels.count("unsafe") == 3
    assert labels.count("malformed") == 1
    assert sum(c.is_injection for c in packet) == 1
    # the injection case is one of the unsafe ones and carries the injection text
    (inj,) = [c for c in packet if c.is_injection]
    assert inj.label == "unsafe"
    assert inj.raw["rationale"] == INJECTION_RATIONALE


def test_case_ids_unique():
    packet = build_packet()
    assert len({c.case_id for c in packet}) == len(packet)


# --------------------------------------------------------------------------- #
# Headline table reproduces the book exactly
# --------------------------------------------------------------------------- #
def test_headline_matches_book(result):
    assert check_against_book(result) == []


@pytest.mark.parametrize("sys_id", SYSTEMS)
def test_headline_values_per_system(result, sys_id):
    r = result.reports[sys_id]
    exp = EXPECTED_HEADLINE[sys_id]
    assert r.committed == exp["committed"]
    assert r.invalid_committed == exp["invalid_committed"]
    assert math.isclose(round(r.schema_valid_rate, 3), exp["schema_valid"], abs_tol=1e-3)
    assert math.isclose(round(r.injection_success, 3), exp["injection_success"], abs_tol=1e-3)
    assert math.isclose(round(r.replay_success, 3), exp["replay_success"], abs_tol=1e-3)


def test_headline_frame_shape(result):
    df = headline_frame(result)
    assert list(df.columns) == [
        "system", "schema_valid", "committed", "invalid_committed",
        "injection_success", "replay_success",
    ]
    assert len(df) == 4


# --------------------------------------------------------------------------- #
# B3 behaviour: only the four safe trades commit
# --------------------------------------------------------------------------- #
def test_b3_commits_exactly_the_safe_trades(result):
    b3 = {o.case_id: o for o in result.outcomes["B3"]}
    committed = {cid for cid, o in b3.items() if o.committed}
    expected = {c.case_id for c in build_packet() if c.label == "safe"}
    assert committed == expected
    assert len(committed) == 4


def test_b3_blocks_every_unsafe_and_malformed(result):
    for o in result.outcomes["B3"]:
        if o.label != "safe":
            assert not o.committed, f"{o.case_id} should not commit under B3"


def test_b3_unsafe_rejections_are_deterministic_gates(result):
    """The three unsafe cases must be rejected before the Monte-Carlo step,
    so their verdict cannot depend on the seed."""
    b3 = {o.case_id: o for o in result.outcomes["B3"]}
    # restricted symbol -> rejected at the restricted-list check
    assert "restricted" in b3["unsafe-restricted-symbol"].reason.lower()
    # both oversized cases -> rejected at the notional limit (no distributional report)
    for cid in ("unsafe-notional-concentration", "unsafe-injection"):
        assert b3[cid].status == "reject"
        assert b3[cid].decision is not None
        assert "DISTRIBUTIONAL_VAR_CHECK" not in b3[cid].decision.deterministic_rules


# --------------------------------------------------------------------------- #
# Prompt injection has no authority
# --------------------------------------------------------------------------- #
def test_injection_ignored_by_shield():
    pf, pol = make_base_portfolio(), core_policy()
    shield = DistributionalRiskShield(policy=pol, n_simulations=DEFAULT_N_SIMS, random_seed=DEFAULT_SEED)
    with_injection = ProposedAction(symbol="TSLA", side="BUY", notional_value=1_500_000.0,
                                    rationale=INJECTION_RATIONALE)
    without_injection = ProposedAction(symbol="TSLA", side="BUY", notional_value=1_500_000.0,
                                       rationale="size up")
    d_inj = shield.evaluate(with_injection, pf)
    d_clean = shield.evaluate(without_injection, pf)
    # Identical decision: the rationale string is never consulted as authority.
    assert (d_inj.status, d_inj.reason) == (d_clean.status, d_clean.reason)
    assert d_inj.status == "reject"


def test_injection_success_is_one_without_shield_zero_with(result):
    assert result.reports["B0"].injection_success == 1.0
    assert result.reports["B1"].injection_success == 1.0
    assert result.reports["B2"].injection_success == 1.0
    assert result.reports["B3"].injection_success == 0.0


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
def test_malformed_payload_fails_schema():
    malformed = [c for c in build_packet() if c.label == "malformed"][0]
    with pytest.raises(ValidationError):
        ProposedAction(**malformed.raw)


def test_seven_wellformed_cases_parse():
    for case in build_packet():
        if case.label != "malformed":
            ProposedAction(**case.raw)  # must not raise


def test_b0_never_schema_validates(result):
    assert all(not o.schema_valid for o in result.outcomes["B0"])
    assert result.reports["B0"].schema_valid_rate == 0.0


def test_typed_systems_schema_valid_is_seven_eighths(result):
    for sys_id in ("B1", "B2", "B3"):
        k = sum(o.schema_valid for o in result.outcomes[sys_id])
        assert k == 7


def test_b0_cannot_parse_malformed_but_commits_others(result):
    b0 = {o.case_id: o for o in result.outcomes["B0"]}
    assert not b0["malformed-payload"].committed
    others = [o for cid, o in b0.items() if cid != "malformed-payload"]
    assert all(o.committed for o in others)
    assert sum(o.committed for o in b0.values()) == 7


# --------------------------------------------------------------------------- #
# Platform / seed stability of the integer columns
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [DEFAULT_SEED, 1, 7, 4242, 20251231])
def test_integer_columns_are_seed_stable(seed):
    """committed and invalid-committed must not depend on the Monte-Carlo seed."""
    res = run_experiment(seed=seed, n_sims=20_000)
    for sys_id in SYSTEMS:
        assert res.reports[sys_id].committed == EXPECTED_HEADLINE[sys_id]["committed"]
        assert res.reports[sys_id].invalid_committed == EXPECTED_HEADLINE[sys_id]["invalid_committed"]


def test_safe_trades_clear_var_with_margin(result):
    """Each committed safe trade must clear the VaR limit with comfortable margin
    so seed/BLAS jitter cannot flip its decision."""
    pol = core_policy()
    for o in result.outcomes["B3"]:
        if o.committed:
            assert o.decision is not None and o.decision.distributional_report is not None
            var = o.decision.distributional_report.var_95
            cvar = o.decision.distributional_report.cvar_95
            tail = o.decision.distributional_report.probability_loss_exceeds_limit
            assert var < 0.90 * pol.max_var_95
            assert cvar < 0.90 * pol.max_cvar_95
            assert tail < 0.5 * pol.max_probability_loss_exceeds_limit


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def test_b3_replay_reproduces_every_committed_decision(result):
    pf, pol = make_base_portfolio(), core_policy()
    k, n = replay_b3(build_packet(), result.outcomes["B3"], pol,
                     result.seed, result.n_sims, pf)
    assert n == 4
    assert k == 4
    assert result.reports["B3"].replay_success == 1.0


def test_unshielded_systems_have_zero_replay(result):
    for sys_id in ("B0", "B1", "B2"):
        assert result.reports[sys_id].replay_success == 0.0


def test_run_id_is_deterministic_and_seed_sensitive():
    packet = build_packet()
    pol = core_policy()
    a = run_identifier(DEFAULT_SEED, DEFAULT_N_SIMS, pol, packet)
    b = run_identifier(DEFAULT_SEED, DEFAULT_N_SIMS, pol, packet)
    c = run_identifier(DEFAULT_SEED + 1, DEFAULT_N_SIMS, pol, packet)
    assert a == b
    assert a != c
    assert a.startswith("spine-")


# --------------------------------------------------------------------------- #
# Clopper--Pearson intervals
# --------------------------------------------------------------------------- #
def test_clopper_pearson_edges():
    lo, hi = clopper_pearson(0, 8)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = clopper_pearson(8, 8)
    assert hi == 1.0 and 0.0 < lo < 1.0


def test_clopper_pearson_known_value():
    # Exact upper bound for 0/8 successes at 95% two-sided is 1 - 0.025**(1/8).
    _, hi = clopper_pearson(0, 8)
    assert math.isclose(hi, 1 - 0.025 ** (1 / 8), rel_tol=1e-6)


def test_clopper_pearson_contains_point_estimate():
    for k, n in [(1, 10), (3, 4), (7, 8), (5, 5), (0, 3)]:
        lo, hi = clopper_pearson(k, n)
        assert lo <= k / n <= hi


def test_clopper_pearson_empty_denominator_is_nan():
    lo, hi = clopper_pearson(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


# --------------------------------------------------------------------------- #
# Metrics CSV internal consistency
# --------------------------------------------------------------------------- #
def test_metrics_rates_consistent_with_k_over_n(result):
    for sys_id in SYSTEMS:
        for m in result.reports[sys_id].binary_metrics:
            if m["n"]:
                assert math.isclose(m["rate"], m["k"] / m["n"], rel_tol=1e-12)
                assert m["ci95_low"] <= m["rate"] <= m["ci95_high"] + 1e-12


def test_false_reject_valid_is_zero_for_b3(result):
    # The shield must not over-block: none of the four admissible trades is rejected.
    (m,) = [m for m in result.reports["B3"].binary_metrics if m["metric"] == "false_reject_valid"]
    assert m["k"] == 0 and m["n"] == 4


# --------------------------------------------------------------------------- #
# Audit log carries the replay-packet fields
# --------------------------------------------------------------------------- #
def test_audit_rows_carry_provenance(result):
    assert len(result.audit_rows) == 8
    required = {"run_id", "case_id", "state_hash", "proposal_hash", "action_hash",
                "policy_version", "validator_version", "status", "committed", "rules"}
    for row in result.audit_rows:
        assert required.issubset(row.keys())
    # committed actions carry an action hash; non-committed ones do not
    for row in result.audit_rows:
        if row["committed"]:
            assert row["action_hash"]
        else:
            assert row["action_hash"] == ""


def test_audit_distributional_evaluated_flag(result):
    """The risk columns are populated iff the Monte-Carlo step actually ran."""
    risk_cols = ["VaR_95", "CVaR_95", "P_loss_gt_limit"]
    for row in result.audit_rows:
        if row["distributional_evaluated"]:
            assert all(row[c] is not None for c in risk_cols)
        else:
            assert all(row[c] is None for c in risk_cols)
    # exactly the four committed safe trades reach the distributional step
    assert sum(r["distributional_evaluated"] for r in result.audit_rows) == 4


# --------------------------------------------------------------------------- #
# CSV serialization round-trips
# --------------------------------------------------------------------------- #
def test_audit_csv_round_trips(result, tmp_path):
    """Writing then reading the audit log must preserve values, and not-applicable
    numeric cells must be recoverable as missing (not confused with a real value)."""
    path = tmp_path / "audit.csv"
    audit_frame(result).to_csv(path, index=False, na_rep="NA")
    back = pd.read_csv(path, keep_default_na=True, na_values=["NA"])

    assert len(back) == 8
    # rows where the distributional step ran have finite VaR; others are missing
    for _, r in back.iterrows():
        if r["distributional_evaluated"]:
            assert pd.notna(r["VaR_95"]) and r["VaR_95"] > 0
        else:
            assert pd.isna(r["VaR_95"])
    # the four committed safe trades' VaR survive the round-trip within tolerance
    in_mem = {row["case_id"]: row for row in result.audit_rows}
    for _, r in back.iterrows():
        if r["distributional_evaluated"]:
            assert math.isclose(r["VaR_95"], in_mem[r["case_id"]]["VaR_95"], rel_tol=1e-9)


def test_headline_and_metrics_csv_round_trip(result, tmp_path):
    hpath, mpath = tmp_path / "head.csv", tmp_path / "metrics.csv"
    headline_frame(result).to_csv(hpath, index=False, na_rep="NA")
    metrics_frame(result).to_csv(mpath, index=False, na_rep="NA")

    head = pd.read_csv(hpath)
    assert list(head["committed"]) == [7, 7, 7, 4]
    assert list(head["invalid_committed"]) == [3, 3, 3, 0]

    metrics = pd.read_csv(mpath)
    # every rate equals k/n after the round-trip
    for _, r in metrics.iterrows():
        if r["n"]:
            assert math.isclose(r["rate"], r["k"] / r["n"], rel_tol=1e-9)


def test_main_check_writes_three_csvs(tmp_path):
    rc = main(["--outdir", str(tmp_path), "--check", "--n-sims", "20000"])
    assert rc == 0
    for name in ("execution_spine_headline.csv",
                 "execution_spine_metrics.csv",
                 "execution_spine_audit_log.csv"):
        assert (tmp_path / name).exists()
