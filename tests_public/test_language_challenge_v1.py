from copy import deepcopy
import json

import pytest

from scripts.language_challenge_v1 import (
    BOUNDARY, MODES, OUTPUT, authored_items, digest, frame, score,
    validate_items, verify,
)


def test_authored_boundary_has_balanced_families_and_literal_value_support():
    items = validate_items(authored_items())
    assert len(items) == 24
    assert len({item["input"] for item in items}) == 24
    for item in items:
        for target, spans in zip(item["expected"], item["target_support_spans"]):
            a, b = spans["subject"]
            assert item["input"][a:b].casefold().replace(" ", "_") == target["slots"]["subject"]
            a, b = spans["number"]
            assert int(item["input"][a:b]) == target["slots"]["value"]


def test_missing_source_support_is_rejected():
    items = deepcopy(authored_items())
    items[0]["expected"][0]["slots"]["value"] = 999999
    with pytest.raises(ValueError, match="literal source support"):
        validate_items(items)


def test_abstaining_on_positive_target_is_not_success():
    target = frame("GOAL", "latency", 3)
    result = score([target], [])
    assert not result["correct"]
    assert result["missing"] == [target]


def test_ambiguous_guess_is_unsupported():
    target = frame("GOAL", "latency", 3)
    assert score([], [target])["unsupported"] == [target]
    assert score([], [])["correct"]


def test_exact_multiset_does_not_ignore_duplicates_or_wrong_numbers():
    target = frame("GOAL", "latency", 3)
    assert not score([target], [target, target])["correct"]
    assert not score([target], [frame("GOAL", "latency", 4)])["correct"]


def test_order_invariance_and_negation_pair_targets():
    items = authored_items()
    order = [i for i in items if i["pair_id"] == "order"]
    assert score(order[0]["expected"], list(reversed(order[1]["expected"])))["correct"]
    negation = [i for i in items if i["pair_id"] == "negation-number"]
    assert not score(negation[0]["expected"], negation[1]["expected"])["correct"]


def test_frozen_challenge_and_evidence_identities():
    assert digest(BOUNDARY / "protocol.json") == "8c777a15aaefefab25a416746ce43c96d3c139e4419761437f9b1f3a19a9878e"
    assert digest(BOUNDARY / "suite.json") == "0fc1a03e07cfdaeb93d265d19ebe97592a36fdef67560ff7ce3346ea4f2ff52e"
    assert digest(OUTPUT / "inventory.json") == "bc4060dd8ef283f77700754b75bf3cd3429fd6d6e6855b5758c3524d73c0b138"
    assert verify()["status"] == "PASS"


def test_raw_reports_recompute_without_a_model():
    for mode in MODES:
        report = json.loads((OUTPUT / f"{mode}.json").read_text(encoding="utf-8"))
        assert len(report["records"]) == 64
        assert len(report["raw_failures"]) == 11
        for row in report["records"]:
            result = score(row["expected"], row["output"]["predicted"])
            assert all(row[key] == value for key, value in result.items())
        assert report["families"]["retention"]["correct"] == 40
        assert report["families"]["resolved_reference"]["correct"] == 0
        assert report["families"]["unfamiliar_phrasing"]["correct"] == 0


def test_no_training_or_activation_and_replay_receipt():
    protocol = json.loads((BOUNDARY / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["training"] == "NONE"
    assert protocol["activation"] == "NONE"
    assert protocol["modes"] == list(MODES)
    receipt = json.loads((OUTPUT.parent / "language-challenge-v1-replay.json").read_text(encoding="utf-8"))
    assert receipt["inference_replayed"] is True
    assert receipt["status"] == "PASS"
