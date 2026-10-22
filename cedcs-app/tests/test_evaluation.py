"""The evaluation harness itself: the synthetic world is reproducible, the evaluator judges against the true state, the statistics are
right, and the property the paper rests on (the gate never recommends a hospital reported to lack a REQUIRED capability) holds."""

from __future__ import annotations

import random

import pytest

from deterministic_core.capability_map import evaluate_capability
from evaluation import experiments, metrics, policies, world
from evaluation.policies import triage_for


@pytest.fixture(autouse=True)
def _fresh_cache():
    policies.clear_cache()
    yield
    policies.clear_cache()


def _snapshot(hs):
    return [(h.hid, h.archetype, round(h.lat, 6), sorted((k, str(v)) for k, v in h.true.items()),
             sorted((k, str(r.value), r.updated_at.isoformat(), r.source) for k, r in h.reported.items())) for h in hs]


def test_a_world_is_reproducible_and_seeds_differ():
    assert _snapshot(world.build_world(7)) == _snapshot(world.build_world(7))
    assert _snapshot(world.build_world(7)) != _snapshot(world.build_world(8))
    assert [c.text for c in world.build_cases(7, 30)] == [c.text for c in world.build_cases(7, 30)]


def test_reports_are_perturbed_truth_more_so_when_older():
    rng = random.Random(1)
    v = {"total": 10, "available": 5}
    assert world._perturb("icu_beds", v, 0.0, rng) == v  # an instant-old report is the truth

    def wrong_rate(age):
        r = random.Random(2)
        return sum(world._perturb("emergency_beds", v, age, r)["available"] != 5 for _ in range(400)) / 400

    assert wrong_rate(0.5) < wrong_rate(4.0) < wrong_rate(24.0) + 0.05 and wrong_rate(12.0) > wrong_rate(0.5)


def test_fixed_age_worlds_have_that_age():
    hs = world.build_world(3, fixed_age_h=4.0)
    ages = [(world.T0 - r.updated_at).total_seconds() / 3600 for h in hs for r in h.reported.values()]
    assert 3.2 <= min(ages) and max(ages) <= 4.8


def test_truth_is_judged_on_the_true_state_not_the_report():
    h = world.build_world(5)[0]
    h.true["department_CARDIOLOGY"] = {"active": True}
    h.reported["department_CARDIOLOGY"] = h.reported.get("department_CARDIOLOGY") or next(iter(h.reported.values())).model_copy(update={"resource_key": "department_CARDIOLOGY"})
    h.reported["department_CARDIOLOGY"] = h.reported["department_CARDIOLOGY"].model_copy(update={"value": {"active": False}})
    h.__dict__.pop("_truth", None)
    assert world.true_status(h, "CARDIOLOGY") == "PRESENT"
    assert evaluate_capability("CARDIOLOGY", policies.scored_records(h, True)).status == "ABSENT"  # what the system believes


def test_the_oracle_is_always_appropriate_and_b1_is_the_nearest_hospital():
    hs = world.build_world(11)
    for c in world.build_cases(11, 60):
        o = policies.oracle(c, hs)
        j = metrics.judge(c, o, hs)
        assert (not j["avoidable"] and not j["issued"]) or j["appropriate"]
        b1 = policies.b1_nearest(c, hs)
        near = min((x for x in hs if world.distance_km(c, x) <= policies.WIDE_KM), key=lambda x: world.distance_km(c, x))
        assert b1.hid == near.hid


def test_the_gate_never_recommends_a_hospital_reported_to_lack_a_required_capability():
    """The paper's safety property, checked over many random worlds, loads and corrupted triages."""
    checked = 0
    for load in ("relaxed", "normal", "crowded"):
        for seed in range(4):
            hs = world.build_world(100 + seed, load=load)
            rng = random.Random(seed)
            for c in world.build_cases(100 + seed, 60):
                tr = triage_for(c, drop_required=rng.choice([0, 0, 1]), priority=rng.choice([None, "LOW"]), rng=rng)
                d = policies.cedcs(c, hs, tr)
                if d.hid is None:
                    continue
                h = next(x for x in hs if x.hid == d.hid)
                for cap in tr.required_capabilities:
                    assert evaluate_capability(cap, policies.scored_records(h, True)).status != "ABSENT", (c.cid, cap)
                checked += 1
            policies.clear_cache()
    assert checked > 500


def test_widening_only_happens_when_fewer_than_three_are_eligible_nearby():
    hs = world.build_world(21)
    for c in world.build_cases(21, 80):
        d = policies.cedcs(c, hs, triage_for(c))
        if d.rung == "widened":
            near = [h for h in hs if world.distance_km(c, h) <= policies.NEAR_KM]
            from deterministic_core.eligibility import filter_eligible
            from deterministic_core.requirements import compute_requirements

            cands = [policies._candidate(h, c) for h in near]
            el = filter_eligible(cands, compute_requirements(triage_for(c)), {x.hospital_id: policies.scored_records(next(h for h in hs if h.hid == x.hospital_id)) for x in cands})
            assert len(el.eligible) < policies.MIN_ELIGIBLE


def test_a_handoff_is_recorded_when_nothing_is_eligible():
    hs = world.build_world(31)
    experiments._degrade(hs, 1.0, random.Random(1))  # every key capability removed everywhere
    c = next(x for x in world.build_cases(31, 50) if x.template.name == "cardiac")
    d = policies.cedcs(c, hs, triage_for(c))
    assert d.hid is None and d.rung == "handoff"
    j = metrics.judge(c, d, hs)
    assert j["handoff"] and j["appropriate"] is None and not j["avoidable"]


def test_statistics():
    lo, hi = metrics.wilson(0, 100)
    assert lo == 0.0 and 0.03 < hi < 0.05
    lo, hi = metrics.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert metrics.mcnemar_exact([True] * 10 + [False] * 10, [False] * 10 + [True] * 10) == 1.0  # symmetric disagreement
    assert metrics.mcnemar_exact([True] * 12 + [False] * 8, [False] * 20) < 0.01  # one-sided disagreement
    m, lo, hi = metrics.mean_ci([1, 2, 3, 4, 5])
    assert m == 3 and lo <= m <= hi


def test_a_small_end_to_end_run_writes_the_expected_files(tmp_path):
    from evaluation import run

    run.main(["--worlds", "1", "--cases", "12", "--out", str(tmp_path)])
    for f in ("RESULTS.md", "results.json", "e1_main_normal.csv", "e2_freshness.csv", "e3_safety.csv", "e4_noise.csv", "e5_weights.csv", "e6_escalation.csv", "figures/e1_undertriage.png"):
        assert (tmp_path / f).exists(), f
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "## E1. Main comparison" in text and "## E6." in text and "CEDCS" in text


def test_b4_has_no_gate_of_any_kind_so_it_can_pick_a_hospital_reported_to_lack_a_capability():
    """B4 must not be silently gated (the engine's final re-verification is a gate): in a crowded city it must sometimes violate."""
    viol = 0
    for seed in range(6):
        hs = world.build_world(200 + seed, load="crowded")
        for c in world.build_cases(200 + seed, 80):
            d = policies.b4_soft_ranking(c, hs, triage_for(c))
            if d.hid is None:
                continue
            h = next(x for x in hs if x.hid == d.hid)
            viol += any(evaluate_capability(cap, policies.scored_records(h, True)).status == "ABSENT" for cap in c.template.required)
        policies.clear_cache()
    assert viol > 0
