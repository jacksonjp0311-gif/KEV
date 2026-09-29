from copy import deepcopy

import pytest

from scripts.language_challenge_v1 import authored_items, frame, score, validate_items


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
