from __future__ import annotations

import copy

import pytest

from kev.actions import ToolRegistry, ToolSpec, verify_receipt
from kev.uc51a3.alive import AliveRuntime


def test_runtime_plans_from_explicit_state_and_persists_receipt_observation(tmp_path):
    registry = ToolRegistry(
        [
            ToolSpec(
                name="measure.temperature",
                handler=lambda data: {"temperature": data["expected"]},
                input_schema={
                    "type": "object",
                    "properties": {"expected": {"type": "integer"}},
                    "required": ["expected"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"temperature": {"type": "integer"}},
                    "required": ["temperature"],
                    "additionalProperties": False,
                },
            )
        ]
    )
    runtime = AliveRuntime(
        tmp_path / "state",
        model_path=tmp_path / "intentionally-missing.pt",
        tool_registry=registry,
    )
    runtime.store.add_frames(
        [
            {
                "id": "prediction-temperature",
                "kind": "PREDICTION",
                "relation": "EXPECTED_VALUE",
                "slots": {
                    "subject": {"type": "text", "value": "temperature"},
                    "operator": {"type": "operator", "value": "=="},
                    "expected": {"type": "number", "value": 21},
                },
            },
            {
                "id": "goal-measure-temperature",
                "kind": "GOAL",
                "relation": "ACHIEVE",
                "slots": {},
                "action": {
                    "tool": "measure.temperature",
                    "input": {"expected": 21},
                },
            },
        ]
    )

    plans = runtime.plan_actions()
    assert len(plans) == 1
    outcome = runtime.execute_action(plans[0])

    assert verify_receipt(outcome["receipt"])
    assert outcome["observation"]["status"] == "VERIFIED"
    assert (
        outcome["observation"]["provenance"]["receipt_hash"]
        == outcome["receipt"]["receipt_hash"]
    )
    assert outcome["prediction_scores"][0]["status"] == "CONFIRMED"

    state = runtime.store.read()
    assert state["observations"][-1]["id"] == outcome["observation"]["id"]
    assert (
        state["prediction_scores"][-1]["receipt_hash"]
        == outcome["receipt"]["receipt_hash"]
    )
    assert runtime.store.verify_ledger()["valid"] is True


def test_runtime_default_tool_registry_is_empty(tmp_path):
    runtime = AliveRuntime(
        tmp_path / "state",
        model_path=tmp_path / "intentionally-missing.pt",
    )
    assert runtime.tools.names() == ()
    rejected = runtime.execute_action("undeclared.tool", {})
    assert rejected["receipt"]["status"] == "REJECTED"
    assert rejected["observation"] is None
    assert runtime.store.verify_ledger()["valid"] is True


def test_store_rejects_a_forged_verified_observation(tmp_path):
    runtime = AliveRuntime(
        tmp_path / "state",
        model_path=tmp_path / "intentionally-missing.pt",
    )
    with pytest.raises(ValueError, match="ledger-backed tool receipt"):
        runtime.store.add_frames(
            [
                {
                    "kind": "OBSERVATION",
                    "relation": "TOOL_RESULT",
                    "slots": {"result": {"type": "json", "value": {"ok": True}}},
                    "status": "VERIFIED",
                    "source": {"type": "TOOL", "tool": "forged.tool"},
                    "receipt_hash": "sha256:" + "0" * 64,
                }
            ]
        )
    assert runtime.store.read()["observations"] == []


@pytest.mark.parametrize("tamper", ["tool", "output", "receipt_reference"])
def test_store_rejects_receipt_replay_for_a_different_observation(tmp_path, tamper):
    registry = ToolRegistry(
        [
            ToolSpec(
                name="measure.temperature",
                handler=lambda _data: {"temperature": 21},
                output_schema={
                    "type": "object",
                    "properties": {"temperature": {"type": "integer"}},
                    "required": ["temperature"],
                    "additionalProperties": False,
                },
            )
        ]
    )
    runtime = AliveRuntime(
        tmp_path / "state",
        model_path=tmp_path / "intentionally-missing.pt",
        tool_registry=registry,
    )
    outcome = runtime.execute_action("measure.temperature", {})
    forged = copy.deepcopy(outcome["observation"])
    forged["id"] = f"forged-{tamper}"

    if tamper == "tool":
        forged["tool"] = "attacker.substitute"
        forged["source"]["tool"] = "attacker.substitute"
        forged["provenance"]["source_id"] = "attacker.substitute"
    elif tamper == "output":
        forged["value"] = {"temperature": 999}
        forged["slots"]["result"]["value"] = {"temperature": 999}
    else:
        forged["provenance"]["receipt_hash"] = "sha256:" + "f" * 64

    with pytest.raises(ValueError, match="exactly matching"):
        runtime.store.add_frames([forged])

    assert len(runtime.store.read()["observations"]) == 1


def test_store_rejects_duplicate_use_of_an_exact_success_receipt(tmp_path):
    registry = ToolRegistry(
        [
            ToolSpec(
                name="measure.temperature",
                handler=lambda _data: {"temperature": 21},
                output_schema={
                    "type": "object",
                    "properties": {"temperature": {"type": "integer"}},
                    "required": ["temperature"],
                    "additionalProperties": False,
                },
            )
        ]
    )
    runtime = AliveRuntime(
        tmp_path / "state",
        model_path=tmp_path / "intentionally-missing.pt",
        tool_registry=registry,
    )
    outcome = runtime.execute_action("measure.temperature", {})

    with pytest.raises(ValueError, match="already been consumed"):
        runtime.store.add_frames([copy.deepcopy(outcome["observation"])])

    assert len(runtime.store.read()["observations"]) == 1
