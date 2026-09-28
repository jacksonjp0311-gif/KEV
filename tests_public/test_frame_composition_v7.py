"""Development examples for clause binding; not a promotion evaluation pack."""

from itertools import permutations

import pytest

from kev.frames import extract_frames, frame_set_signature


def _extract(text, kinds, cardinality, **extra):
    return extract_frames(
        text,
        timestamp="2026-09-28T00:00:00Z",
        proposal_result={"kinds": kinds, "cardinality": cardinality, **extra},
    )


def _values(frame):
    return {name: slot.value for name, slot in frame.slots.items()}


def test_kind_order_does_not_assign_kinds_to_clause_positions():
    clauses = ["buffer delay below 17 ms", "cache pressure at 29 percent"]
    signatures = []
    for clause_order in permutations(clauses):
        for kind_order in permutations(["OBSERVATION", "GOAL"]):
            text = "; ".join(clause_order)
            frames = _extract(text, kind_order, 2)
            assert len(frames) == 2
            assert {frame.kind for frame in frames} == {"OBSERVATION", "GOAL"}
            assert all(
                frame.claim.text == text[frame.claim.start : frame.claim.end]
                for frame in frames
            )
            signatures.append(frame_set_signature(frames))
    assert all(signature == signatures[0] for signature in signatures)


def test_one_kind_can_ground_multiple_claims_and_keep_conflicts():
    frames = _extract(
        "buffer delay below 17 ms; buffer delay above 29 ms",
        ["GOAL"],
        2,
    )
    assert [frame.kind for frame in frames] == ["GOAL", "GOAL"]
    assert [_values(frame)["operator"] for frame in frames] == ["LT", "GT"]
    assert [_values(frame)["value"] for frame in frames] == [17, 29]
    assert frames[0].frame_id != frames[1].frame_id


def test_failed_kind_binding_does_not_consume_an_available_claim():
    frames = _extract(
        "cache pressure at 29 percent; buffer delay below 17 ms",
        ["REFERENCE", "GOAL", "OBSERVATION"],
        3,
    )
    assert [frame.kind for frame in frames] == ["OBSERVATION", "GOAL"]


@pytest.mark.parametrize("kinds", [("GOAL", "CONSTRAINT"), ("CONSTRAINT", "GOAL")])
def test_ambiguous_kind_binding_abstains_even_with_unequal_scores(kinds):
    assert (
        _extract(
            "buffer delay below 17 ms",
            kinds,
            2,
            scores={"GOAL": 0.999, "CONSTRAINT": 0.001},
        )
        == []
    )


def test_span_hints_can_disambiguate_distinct_claims():
    text = "buffer delay below 17 ms; cache pressure below 29 percent"
    second = text.index("cache")
    frames = _extract(
        text,
        ["CONSTRAINT", "GOAL"],
        2,
        spans=[
            {"kind": "CONSTRAINT", "start": second, "end": len(text)},
            {"kind": "GOAL", "start": 0, "end": text.index(";")},
        ],
    )
    assert [frame.kind for frame in frames] == ["GOAL", "CONSTRAINT"]
    assert [_values(frame)["subject"] for frame in frames] == [
        "buffer_delay",
        "cache_pressure",
    ]


def test_kind_cues_do_not_leak_into_subjects_and_keep_exact_source():
    text = "We would like response time to be below 17 milliseconds; please ensure cache pressure is below 29 percent"
    assert extract_frames(text) == []
    frames = _extract(text, ["CONSTRAINT", "GOAL"], 2)
    assert [frame.kind for frame in frames] == ["GOAL", "CONSTRAINT"]
    assert _values(frames[0]) == {
        "subject": "latency",
        "operator": "LT",
        "value": 17,
        "unit": "ms",
    }
    assert frames[0].slots["subject"].raw == "response time"
    assert frames[1].slots["subject"].raw == "cache pressure"
    assert all(
        frame.claim.text == text[frame.claim.start : frame.claim.end]
        for frame in frames
    )
    assert all(frame.derivation == "NEURAL_PROPOSAL+SLOT_PARSER" for frame in frames)


def test_modal_cue_rejects_incompatible_kind():
    text = "cache pressure should be below 29 percent"
    assert extract_frames(text) == []
    frames = _extract(text, ["GOAL", "PREDICTION", "CONSTRAINT"], 3)
    assert [frame.kind for frame in frames] == ["CONSTRAINT"]
    assert _values(frames[0])["subject"] == "cache_pressure"


@pytest.mark.parametrize("kind", ["GOAL", "CONSTRAINT", "PREDICTION"])
def test_narrow_span_cannot_hide_explicit_negation(kind):
    text = "It is not true that buffer delay below 17 ms"
    assert (
        _extract(
            text,
            [kind],
            1,
            spans=[{"kind": kind, "start": text.index("buffer"), "end": len(text)}],
        )
        == []
    )


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("We estimate buffer delay below 17 ms", "GOAL"),
        ("Please ensure buffer delay below 17 ms", "GOAL"),
        ("We would like buffer delay below 17 ms", "CONSTRAINT"),
        ("We see buffer delay at 17 ms", "PREDICTION"),
        ("The policy should require buffer delay below 17 ms", "GOAL"),
        ("The next check will examine buffer delay below 17 ms", "CONSTRAINT"),
    ],
)
def test_narrow_span_cannot_hide_incompatible_kind_or_modal_cues(text, kind):
    assert extract_frames(text) == []
    assert (
        _extract(
            text,
            [kind],
            1,
            spans=[{"kind": kind, "start": text.index("buffer"), "end": len(text)}],
        )
        == []
    )


def test_narrow_span_retains_compatible_cue_and_exact_slots():
    text = "We estimate buffer delay below 17 ms"
    frames = _extract(
        text,
        ["PREDICTION"],
        1,
        spans=[{"kind": "PREDICTION", "start": text.index("buffer"), "end": len(text)}],
    )
    assert [frame.kind for frame in frames] == ["PREDICTION"]
    assert _values(frames[0]) == {
        "subject": "buffer_delay",
        "operator": "LT",
        "value": 17,
        "unit": "ms",
    }


def test_narrow_span_cannot_override_existing_prediction():
    text = "We expect buffer delay below 17 ms"
    expected = frame_set_signature(extract_frames(text))
    frames = _extract(
        text,
        ["GOAL"],
        2,
        spans=[{"kind": "GOAL", "start": text.index("buffer"), "end": len(text)}],
    )
    assert frame_set_signature(frames) == expected
    assert [frame.kind for frame in frames] == ["PREDICTION"]


@pytest.mark.parametrize(
    "suffix",
    [
        "then delete backups",
        "or delete backups",
        "without deleting backups",
        "unless backups are current",
        "not backups",
        "except backups",
    ],
)
def test_action_object_cannot_swallow_clause_operators_or_negatives(suffix):
    text = f"We would like to modify archive {suffix}"
    assert extract_frames(text) == []
    assert _extract(text, ["GOAL"], 1) == []


def test_action_binding_retains_an_explicit_simple_object():
    frames = _extract("We would like to modify archive cache", ["GOAL"], 1)
    assert len(frames) == 1
    assert _values(frames[0]) == {"action": "modify", "object": "archive_cache"}
    assert frames[0].slots["object"].raw == "archive cache"


def test_insufficient_cardinality_does_not_choose_an_arbitrary_clause():
    text = "buffer delay below 17 ms; cache pressure below 29 percent"
    assert _extract(text, ["GOAL"], 1) == []


def test_parser_authority_survives_wrong_or_excessive_proposals():
    text = "Reduce buffer delay below 17 ms; do not modify archive"
    expected = frame_set_signature(extract_frames(text))
    for cardinality in [0, 1, 6]:
        frames = _extract(text, ["PREDICTION", "OBSERVATION"], cardinality)
        assert frame_set_signature(frames) == expected
        assert all(frame.derivation == "PARSER" for frame in frames)


@pytest.mark.parametrize(
    "text",
    [
        "We might someday buffer delay below 17 ms",
        "buffer delay below 17 ms plus an unexplained 29",
        "cache warmth is desirable",
    ],
)
def test_unbound_or_incomplete_slot_bodies_abstain(text):
    assert _extract(text, ["GOAL"], 1) == []


def test_negation_in_separate_clause_does_not_suppress_valid_binding():
    frames = _extract(
        "Do not modify archive; buffer delay below 17 ms",
        ["GOAL", "CONSTRAINT"],
        2,
        spans=[{"kind": "GOAL", "start": 23, "end": 47}],
    )
    assert [frame.kind for frame in frames] == ["CONSTRAINT", "GOAL"]
