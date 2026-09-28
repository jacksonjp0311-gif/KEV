from __future__ import annotations

from datetime import datetime, timezone

from kev.actions import (
    ActionExecutor,
    StateNarrator,
    StatePlanner,
    ToolRegistry,
    ToolSpec,
    score_predictions,
    verify_receipt,
)


INPUT_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"measured": {"type": "integer"}},
    "required": ["measured"],
    "additionalProperties": False,
}


def fixed_clock():
    return datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def test_registry_is_empty_by_default_and_planner_uses_only_structured_goals():
    registry = ToolRegistry()
    assert len(registry) == 0
    registry.register(
        ToolSpec(
            name="measure.echo",
            handler=lambda data: {"measured": data["value"]},
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
        )
    )
    state = {
        "goals": [
            {
                "id": "goal-1",
                "text": "natural language does not execute",
                "action": {"tool": "measure.echo", "input": {"value": 7}},
            }
        ],
        "chat": {"action": {"tool": "measure.echo", "input": {"value": 99}}},
    }
    plans = StatePlanner(registry).plan(state)
    assert len(plans) == 1
    assert plans[0].arguments == {"value": 7}
    assert plans[0].goal_id == "goal-1"


def test_success_writes_canonical_receipt_before_verified_observation():
    ledger = []
    registry = ToolRegistry(
        [
            ToolSpec(
                name="measure.echo",
                handler=lambda data: {"measured": data["value"]},
                input_schema=INPUT_SCHEMA,
                output_schema=OUTPUT_SCHEMA,
            )
        ]
    )
    outcome = ActionExecutor(registry, ledger.append, clock=fixed_clock).execute(
        "measure.echo", {"value": 42}, state={"goals": []}
    )

    assert len(ledger) == 1
    assert verify_receipt(ledger[0])
    assert outcome.receipt["status"] == "SUCCEEDED"
    assert outcome.observation["kind"] == "OBSERVATION"
    assert outcome.observation["status"] == "VERIFIED"
    assert outcome.observation["source"] == {"type": "TOOL", "tool": "measure.echo"}
    assert outcome.observation["receipt_hash"] == ledger[0]["receipt_hash"]


def test_schema_failure_and_tool_failure_are_preserved_without_observations():
    ledger = []

    def broken(_data):
        raise RuntimeError("raw adapter failure")

    registry = ToolRegistry(
        [
            ToolSpec(
                name="measure.broken",
                handler=broken,
                input_schema=INPUT_SCHEMA,
                output_schema=OUTPUT_SCHEMA,
            )
        ]
    )
    executor = ActionExecutor(registry, ledger.append, clock=fixed_clock)
    rejected = executor.execute("measure.broken", {"value": "not-an-integer"})
    failed = executor.execute("measure.broken", {"value": 1})

    assert rejected.receipt["status"] == "REJECTED"
    assert rejected.observation is None
    assert "expected integer" in rejected.receipt["failure"]["message"]
    assert failed.receipt["status"] == "FAILED"
    assert failed.observation is None
    assert failed.receipt["failure"]["message"] == "raw adapter failure"
    assert all(verify_receipt(item) for item in ledger)


def test_production_mutation_is_never_invoked_and_is_receipted():
    called = []
    ledger = []
    registry = ToolRegistry(
        [
            ToolSpec(
                name="production.write",
                handler=lambda data: called.append(data),
                input_schema={"type": "object"},
                output_schema={},
                production_mutation=True,
            )
        ]
    )
    outcome = ActionExecutor(registry, ledger.append, clock=fixed_clock).execute(
        "production.write", {}
    )
    assert called == []
    assert outcome.observation is None
    assert outcome.receipt["failure"]["code"] == "PRODUCTION_MUTATION_FORBIDDEN"
    assert outcome.receipt["production_mutation_requested"] is True
    assert outcome.receipt["production_mutation_performed"] is False
    assert len(ledger) == 1


def test_predictions_are_scored_only_against_later_matching_observations():
    predictions = [
        {
            "id": "prediction-1",
            "kind": "PREDICTION",
            "subject": "latency_ms",
            "operator": "<",
            "value": 50,
            "created_at": "2026-09-28T12:00:00Z",
        }
    ]
    observations = [
        {
            "kind": "OBSERVATION",
            "subject": "latency_ms",
            "value": 20,
            "observed_at": "2026-09-28T11:59:00Z",
        },
        {
            "id": "observation-1",
            "kind": "OBSERVATION",
            "subject": "latency_ms",
            "value": 47,
            "observed_at": "2026-09-28T12:01:00Z",
            "receipt_hash": "sha256:receipt",
        },
    ]
    result = score_predictions(predictions, observations)
    assert result[0]["status"] == "CONFIRMED"
    assert result[0]["score"] == 1.0
    assert result[0]["actual"] == 47


def test_narrator_tags_arbitrary_generation_ungrounded_and_never_writes_facts():
    state = {"facts": {"codename": {"value": "Nova"}}, "goals": []}
    before = {"facts": {"codename": {"value": "Nova"}}, "goals": []}
    narration = StateNarrator().narrate(
        state, generated_text="The deployment succeeded."
    )
    assert narration["grounding"] == "UNGROUNDED"
    assert narration["fact_updates"] == []
    assert state == before
