import pytest

from kev.clause_proposals import ClauseLocalPredictor, locate_proposals


def head(kind, count=1):
    return {"proposals": [{"kind": kind, "score": 0.8}], "cardinality": count}


def test_exact_decimal_offsets_and_repeated_kind():
    text = "We wish queue delay below 3.5 ms; we wish cache delay below 8 ms."
    proposal, trace = locate_proposals(text, lambda _: head("GOAL"))
    assert proposal.cardinality == 2
    assert len(proposal.spans) == 2
    assert all(text[row["start"]:row["end"]] == row["text"] for row in trace)
    assert "3.5" in trace[0]["text"]


def test_local_kinds_cannot_leak_into_other_clause():
    text = "We wish queue delay below 3 ms; we see cache delay is 8 ms."
    predictor = ClauseLocalPredictor(
        lambda clause: head("GOAL" if "wish" in clause else "PREDICTION"), "a" * 64
    )
    result = predictor({"id": "dev", "input": text})
    assert [frame["kind"] for frame in result["predicted"]] == ["GOAL"]
    assert result["score_status"] == "UNCALIBRATED"
    assert result["confidence"] is None


def test_parser_survives_zero_cardinality():
    predictor = ClauseLocalPredictor(lambda _: head("GOAL", 0), "a" * 64)
    result = predictor({"id": "dev", "input": "Keep queue delay below 3 ms."})
    assert len(result["predicted"]) == 1


def test_negative_clause_does_not_create_positive_frame():
    predictor = ClauseLocalPredictor(lambda _: head("OBSERVATION"), "a" * 64)
    result = predictor({"id": "dev", "input": "It is not true that queue delay is 3 ms."})
    assert result["predicted"] == []


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_cardinality_rejected(count):
    with pytest.raises(ValueError, match="cardinality"):
        locate_proposals("hello", lambda _: head("GOAL", count))


def test_zero_cardinality_never_donates_span():
    proposal, _ = locate_proposals("first; second", lambda s: head("GOAL", int(s == "second")))
    assert len(proposal.spans) == 1
    assert proposal.spans[0].start == 7


def test_correct_local_heads_bind_values_not_proposed_slots():
    def proposal(clause):
        value = head("GOAL" if "wish" in clause else "OBSERVATION")
        value["slots"] = {"value": 9999}
        return value

    predictor = ClauseLocalPredictor(proposal, "a" * 64)
    text = "We wish queue delay below 3 ms; we see cache delay is 8 ms."
    result = predictor({"id": "dev", "input": text})
    assert [(f["kind"], f["slots"]["value"]) for f in result["predicted"]] == [
        ("GOAL", 3), ("OBSERVATION", 8),
    ]
    for frame in predictor.trace[0]["frames"]:
        claim = frame["claim"]
        assert text[claim["start"]:claim["end"]] == claim["text"]


def test_empty_text_does_not_call_head():
    def forbidden(_):
        raise AssertionError("head should not run")

    proposal, trace = locate_proposals("  ", forbidden)
    assert proposal.cardinality == 0
    assert trace == []
