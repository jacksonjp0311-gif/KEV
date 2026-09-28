import json

from scripts.clause_local_experiment import BOUNDARY, OUTPUT, ROOT, digest
from scripts.verify_clause_local_evidence import verify


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_primary_inventory_and_frozen_source_closure():
    assert digest(OUTPUT / "inventory.json") == "61f201bfacaa998e32a8f8d1092b1791bcdd6b8efb1505fd2b4c1bed842b8b91"
    inventory = read(OUTPUT / "inventory.json")
    assert digest(ROOT / "scripts/verify_clause_local_evidence.py") == inventory["verifier_sha256"]
    result = verify()
    assert result["status"] == "PASS"
    assert result["inference_replayed"] is False
    assert result["ledger"]["events"] == 2


def test_research_never_claims_production_qualification():
    card = read(OUTPUT / "research-card.json")
    assert card["activation"] == "NONE"
    assert card["hypothesis"] == "NOT_SUPPORTED"
    assert card["production_qualification"] == "NOT_EVALUATED_RUNTIME_NOT_INTEGRATED"
    assert card["score_status"] == "UNCALIBRATED"
    events = [json.loads(line) for line in (OUTPUT / "research-ledger/ledger.jsonl").read_text().splitlines()]
    assert [event["kind"] for event in events] == ["RESEARCH_STARTED", "RESEARCH_COMPLETED"]


def test_posthoc_controls_are_retained_not_hidden():
    directory = ROOT / "experiments/clause-local-v1-controls"
    assert digest(directory / "inventory.json") == "508641585f53ebf1a5a7fba4229476befe1f453946740855292f4c29719245d8"
    for relative, expected in read(directory / "inventory.json")["files"].items():
        assert digest(directory / relative) == expected
    plan = read(directory / "plan.json")
    assert plan["status"] == "POST_HOC_AFTER_PRIMARY_RESULTS_NOT_PREREGISTERED"
    assert digest(ROOT / "scripts/clause_local_controls.py") == plan["source_sha256"]
    summary = read(directory / "summary.json")
    assert summary["reports"]["all_kinds_with_clause_count_budget_no_model"]["scores"]["composition"]["correct"] == 56
    candidate = read(OUTPUT / "candidate-ablations.json")
    assert candidate["local"]["splits"]["composition"]["correct"] == 55
    assert candidate["global"]["splits"]["composition"]["correct"] == 2
    assert candidate["localization"]["correct"] == 79


def test_new_boundary_training_separation_and_replay_receipt():
    suite = read(BOUNDARY / "suite.json")
    protocol = read(BOUNDARY / "protocol.json")
    surfaces = {item["input"].strip().casefold().rstrip(".") for items in suite["splits"].values() for item in items}
    rows = [json.loads(line) for line in (BOUNDARY / "reviewed-lessons.jsonl").read_text().splitlines()]
    assert len(rows) == 128
    for row in rows:
        assert row["status"] == "REVIEWED"
        assert row["text"].strip().casefold().rstrip(".") not in surfaces
        assert not any(term in row["text"].casefold() for term in protocol["reserved_vocabulary"])
    assert read(OUTPUT / "replay-receipt.json")["status"] == "PASS"
