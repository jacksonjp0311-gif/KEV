from __future__ import annotations

import hashlib

import pytest

from kev.frames import (
    BASE_RELATIONS,
    COGNITIVE_KINDS,
    ProposalHeadResult,
    extract_frames,
    frame_set_signature,
)


STAMP = "2026-09-28T12:00:00Z"


def _values(frame):
    return {name: slot.value for name, slot in frame.slots.items()}


def _extract(text, **kwargs):
    return extract_frames(
        text,
        timestamp=STAMP,
        source_id="test-utterance",
        actor="tester",
        **kwargs,
    )


def test_single_frame_has_typed_slots_exact_boundary_and_provenance():
    text = "The last measured latency was 73 ms."
    frames = _extract(text)

    assert len(frames) == 1
    frame = frames[0]
    assert (frame.kind, frame.relation, frame.status) == (
        "OBSERVATION",
        "MEASUREMENT",
        "REPORTED",
    )
    assert _values(frame) == {"subject": "latency", "value": 73, "unit": "ms"}
    assert frame.slots["value"].slot_type == "number"
    assert frame.slots["value"].raw == "73"
    assert text[frame.claim.start : frame.claim.end] == frame.claim.text
    assert frame.provenance.source_type == "language"
    assert frame.provenance.source_id == "test-utterance"
    assert frame.provenance.actor == "tester"
    assert frame.provenance.timestamp == STAMP
    assert (
        frame.provenance.utterance_sha256 == hashlib.sha256(text.encode()).hexdigest()
    )


def test_one_utterance_yields_goal_constraint_and_observation():
    text = (
        "Reduce latency below 50 ms, do not modify production, "
        "and the last measured latency was 73 ms."
    )
    frames = _extract(text)

    assert [frame.kind for frame in frames] == ["GOAL", "CONSTRAINT", "OBSERVATION"]
    assert _values(frames[0]) == {
        "subject": "latency",
        "operator": "LT",
        "value": 50,
        "unit": "ms",
    }
    assert _values(frames[1]) == {
        "action": "modify",
        "object": "production",
        "polarity": False,
        "modality": "FORBIDDEN",
    }
    assert _values(frames[2]) == {"subject": "latency", "value": 73, "unit": "ms"}
    assert [frame.claim.text for frame in frames] == [
        "Reduce latency below 50 ms",
        "do not modify production",
        "the last measured latency was 73 ms",
    ]
    assert all(text[f.claim.start : f.claim.end] == f.claim.text for f in frames)


def test_elliptical_measurement_binds_to_prior_explicit_metric_only():
    text = "Reduce latency below 50 ms, no production mutation, measured 73 ms."
    frames = _extract(text)
    assert [frame.kind for frame in frames] == ["GOAL", "CONSTRAINT", "OBSERVATION"]
    assert _values(frames[-1]) == {"subject": "latency", "value": 73, "unit": "ms"}
    assert frames[-1].claim.text == "measured 73 ms"

    # A number is not self-describing, so no subject is invented in isolation.
    assert _extract("Measured 73 ms.") == []


def test_polite_buried_constraint_is_not_lost():
    text = "Could you please optimize the cache, but please do not modify production?"
    frames = _extract(text)

    assert [frame.kind for frame in frames] == ["GOAL", "CONSTRAINT"]
    constraint = frames[1]
    assert _values(constraint)["action"] == "modify"
    assert _values(constraint)["object"] == "production"
    assert _values(constraint)["polarity"] is False
    assert constraint.claim.text == "please do not modify production"


def test_buried_threshold_and_passive_negation_preserve_two_claims():
    frames = _extract(
        "Please improve the service to keep latency below 45 ms without modifying production"
    )
    assert [frame.kind for frame in frames] == ["GOAL", "CONSTRAINT"]
    assert _values(frames[0])["value"] == 45
    assert _values(frames[1])["polarity"] is False

    passive = _extract(
        "Lower response time under 50 milliseconds; production must not be changed"
    )
    assert frame_set_signature(passive) == frame_set_signature(
        _extract("Reduce latency below 50 ms; do not modify production")
    )


def test_paraphrases_have_the_same_frame_set_semantics():
    direct = _extract("Reduce latency below 50 milliseconds; do not modify production.")
    paraphrase = _extract(
        "Keep response time under 50 ms while leaving production unchanged."
    )

    assert frame_set_signature(direct) == frame_set_signature(paraphrase)


def test_conflicting_constraints_remain_two_separate_claims():
    text = "Do not modify production, but production must be modified."
    frames = _extract(text)

    assert [frame.kind for frame in frames] == ["CONSTRAINT", "CONSTRAINT"]
    assert [_values(frame)["polarity"] for frame in frames] == [False, True]
    assert [_values(frame)["object"] for frame in frames] == [
        "production",
        "production",
    ]
    assert frames[0].claim.end <= frames[1].claim.start


def test_negation_order_and_number_changes_are_semantically_distinct():
    forbidden = _extract("Do not modify production.")[0]
    required = _extract("Production must be modified.")[0]
    assert forbidden.semantic_key() != required.semantic_key()

    forward = _extract("Copy alpha to beta.")[0]
    reverse = _extract("Copy beta to alpha.")[0]
    assert _values(forward) == {"source": "alpha", "destination": "beta"}
    assert _values(reverse) == {"source": "beta", "destination": "alpha"}
    assert forward.semantic_key() != reverse.semantic_key()

    fifty = _extract("Reduce latency below 50 ms.")[0]
    sixty = _extract("Reduce latency below 60 ms.")[0]
    assert _values(fifty)["value"] == 50
    assert _values(sixty)["value"] == 60
    assert fifty.semantic_key() != sixty.semantic_key()

    threshold_constraint = _extract("Latency must be below 50 ms.")[0]
    assert threshold_constraint.relation == "THRESHOLD"
    assert _values(threshold_constraint)["subject"] == "latency"

    percent_goal = _extract("Reduce error rate below 2.5 %.")[0]
    assert _values(percent_goal)["value"] == 2.5
    assert _values(percent_goal)["unit"] == "%"


@pytest.mark.parametrize(
    "text",
    [
        "Do not lower latency below 50 ms",
        "I don't expect latency to be below 50 ms",
        "It is not true that the last measured latency was 73 ms",
        "Do not copy alpha to beta",
    ],
)
def test_explicit_negation_never_leaks_a_positive_frame(text):
    frames = _extract(text)
    positive_kinds = {"GOAL", "PREDICTION", "OBSERVATION", *BASE_RELATIONS}

    assert positive_kinds.isdisjoint(frame.kind for frame in frames)
    assert all(
        text[frame.claim.start : frame.claim.end] == frame.claim.text
        for frame in frames
    )
    assert all(
        frame.provenance.utterance_sha256 == hashlib.sha256(text.encode()).hexdigest()
        for frame in frames
    )

    # A neural proposal cannot re-introduce the positive assertion that the
    # deterministic parser rejected. Exact values and relation identity still
    # remain under parser authority.
    proposed = _extract(
        text,
        proposal_result={
            "kinds": ["GOAL", "PREDICTION", "OBSERVATION", "COPY_VALUE"],
            "cardinality": 6,
            "spans": [
                {"kind": kind, "start": 0, "end": len(text), "relation": "OVERRIDE"}
                for kind in ("GOAL", "PREDICTION", "OBSERVATION", "COPY_VALUE")
            ],
        },
    )
    assert positive_kinds.isdisjoint(frame.kind for frame in proposed)


@pytest.mark.parametrize(
    ("text", "expected_order"),
    [
        (
            "Measured 73 ms, then reduce latency below 50 ms",
            ["OBSERVATION", "GOAL"],
        ),
        (
            "Reduce latency below 50 ms, then measured 73 ms",
            ["GOAL", "OBSERVATION"],
        ),
    ],
)
def test_elliptical_measurement_and_goal_survive_both_clause_orders(
    text, expected_order
):
    frames = _extract(text)

    assert [frame.kind for frame in frames] == expected_order
    observation = next(frame for frame in frames if frame.kind == "OBSERVATION")
    goal = next(frame for frame in frames if frame.kind == "GOAL")
    assert _values(observation) == {"subject": "latency", "value": 73, "unit": "ms"}
    assert _values(goal) == {
        "subject": "latency",
        "operator": "LT",
        "value": 50,
        "unit": "ms",
    }
    assert observation.claim.text.casefold() == "measured 73 ms"
    assert all(
        text[frame.claim.start : frame.claim.end] == frame.claim.text
        for frame in frames
    )


def test_polite_without_ever_constraint_is_preserved_with_exact_claim():
    text = "Please improve the service without ever modifying production"
    frames = _extract(text)

    assert [frame.kind for frame in frames] == ["GOAL", "CONSTRAINT"]
    assert _values(frames[1]) == {
        "action": "modify",
        "object": "production",
        "polarity": False,
        "modality": "FORBIDDEN",
    }
    assert frames[1].claim.text == "without ever modifying production"
    assert text[frames[1].claim.start : frames[1].claim.end] == frames[1].claim.text


@pytest.mark.parametrize(
    ("text", "expected_kind"),
    [
        ("Copy alpha to beta.", "COPY_VALUE"),
        ("Red is obsolete; blue applies now.", "SUPERSEDES"),
        ("The magnitude of -7 is 7.", "MAGNITUDE"),
        ("Negate -7.", "NEGATE"),
        ("Reference run:7.", "REFERENCE"),
        ("Blue is active.", "ACTIVE_SELECTION"),
        ("Run r7 completed.", "RUN_STATUS"),
        ("Receipt rcpt-1 returned 42.", "RECEIPT_VALUE"),
        ("The evidence is consistent.", "EVIDENCE_CONSISTENCY"),
    ],
)
def test_every_existing_base_relation_has_a_parser(text, expected_kind):
    frames = _extract(text)
    assert expected_kind in {frame.kind for frame in frames}
    assert expected_kind in BASE_RELATIONS


def test_prediction_is_a_distinct_cognitive_kind():
    frame = _extract("We expect latency will be 42 ms.")[0]
    assert frame.kind == "PREDICTION"
    assert frame.kind in COGNITIVE_KINDS
    assert _values(frame) == {
        "subject": "latency",
        "operator": "EQ",
        "value": 42,
        "unit": "ms",
    }


def test_neural_head_is_only_an_optional_bounded_proposal():
    calls = []

    def proposal_head(text):
        calls.append(text)
        return ProposalHeadResult(("GOAL",), 1, {"GOAL": 0.8})

    text = "Latency below 50 ms"
    frame = _extract(text, proposal_head=proposal_head)[0]

    assert calls == [text]
    assert frame.kind == "GOAL"
    assert frame.derivation == "NEURAL_PROPOSAL+SLOT_PARSER"
    assert frame.proposal_score == pytest.approx(0.8)
    assert _values(frame) == {
        "subject": "latency",
        "operator": "LT",
        "value": 50,
        "unit": "ms",
    }
    assert frame.claim.text == text


def test_neural_kind_without_parser_backed_slots_does_not_become_state():
    frames = _extract(
        "Cache warmth is desirable",
        proposal_result={"intent": "GOAL", "score": 0.99},
    )
    assert frames == []


def test_proposal_span_cannot_override_parser_assigned_relation():
    text = "Latency below 50 ms"
    frames = _extract(
        text,
        proposal_result={
            "kinds": ["GOAL"],
            "cardinality": 1,
            "spans": [
                {
                    "kind": "GOAL",
                    "start": 0,
                    "end": len(text),
                    "score": 0.9,
                    "relation": "UNTRUSTED_OVERRIDE",
                }
            ],
        },
    )

    assert len(frames) == 1
    assert frames[0].relation == "THRESHOLD"
    assert frames[0].claim.text == text


def test_parser_frames_are_never_deleted_by_a_wrong_cardinality_proposal():
    text = "Reduce latency below 50 ms and do not modify production."
    frames = _extract(
        text,
        proposal_result={"kinds": ["GOAL"], "cardinality": 1},
    )
    assert [frame.kind for frame in frames] == ["GOAL", "CONSTRAINT"]
    assert all(frame.derivation == "PARSER" for frame in frames)


def test_only_receipted_tool_observations_can_be_verified():
    text = "Measured latency was 73 ms."
    language = _extract(text)[0]
    unreceipted_tool = _extract(text, source_type="tool")[0]
    receipted_tool = _extract(
        text,
        source_type="tool",
        receipt_hash="a" * 64,
    )[0]

    assert language.status == "REPORTED"
    assert unreceipted_tool.status == "REPORTED"
    assert receipted_tool.status == "VERIFIED"
    assert receipted_tool.provenance.receipt_hash == "a" * 64


def test_dict_form_keeps_kind_relation_slots_claim_and_provenance_separate():
    frame = _extract(
        "Reduce latency below 50 ms.",
        model_hash="b" * 64,
    )[0]
    record = frame.to_dict()

    assert record["schema"] == "kev.frame.v1"
    assert record["kind"] == "GOAL"
    assert record["relation"] == "THRESHOLD"
    assert record["slots"]["value"] == {
        "type": "number",
        "value": 50,
        "raw": "50",
    }
    assert record["claim"]["text"] == "Reduce latency below 50 ms"
    assert record["provenance"]["model_hash"] == "b" * 64
